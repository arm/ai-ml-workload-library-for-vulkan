/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "standalone_compute/test_standalone_compute_utils.hpp"
#include "standalone_data_graph/test_standalone_data_graph_utils.hpp"
#include "test_composed_utils.hpp"

#include "internal/workload_builder.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

using mlworkloadlib::detail::Resource;
using mlworkloadlib::detail::WorkloadBuilder;

class ComposedSessionExecutionTest : public RuntimeSessionExecutionTest {};

ModuleImplementation makeSpirvModule(std::vector<uint32_t> spirv) {
    ModuleImplementation module;
    module.codeKind = ModuleCodeKind::Spirv;
    module.spirv = std::move(spirv);
    return module;
}

std::vector<int8_t> makeSmallTensorInput(const std::vector<int64_t> &shape, int8_t seed) {
    std::vector<int8_t> input(Tensor::numElements(shape));
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<int8_t>((static_cast<int32_t>(index % 17) + seed) % 31);
    }
    return input;
}

std::vector<int8_t> addTensors(const std::vector<int8_t> &lhs, const std::vector<int8_t> &rhs) {
    std::vector<int8_t> result(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                   [](int8_t lhsValue, int8_t rhsValue) { return static_cast<int8_t>(lhsValue + rhsValue); });
    return result;
}

std::vector<int8_t> subtractTensors(const std::vector<int8_t> &lhs, const std::vector<int8_t> &rhs) {
    std::vector<int8_t> result(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                   [](int8_t lhsValue, int8_t rhsValue) { return static_cast<int8_t>(lhsValue - rhsValue); });
    return result;
}

template <typename RecordCommands>
void recordAndSubmitCommands(const vk::raii::Device &device, const vk::raii::Queue &queue, uint32_t queueFamilyIndex,
                             RecordCommands recordCommands) {
    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    auto commandBuffer =
        std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    recordCommands(*commandBuffer);
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);
}

void insertAppTensorHandoffBarrier(const vk::raii::Device &device, vk::CommandBuffer commandBuffer,
                                   vk::TensorARM tensor, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) {
    vk::TensorMemoryBarrierARM tensorBarrier;
    tensorBarrier.srcStageMask = srcStage;
    tensorBarrier.srcAccessMask = srcAccess;
    tensorBarrier.dstStageMask = dstStage;
    tensorBarrier.dstAccessMask = dstAccess;
    tensorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    tensorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    tensorBarrier.tensor = tensor;

    const vk::TensorDependencyInfoARM tensorDependencyInfo(1, &tensorBarrier);
    vk::DependencyInfo dependencyInfo;
    dependencyInfo.pNext = &tensorDependencyInfo;

    const auto *const dispatcher = device.getDispatcher();
    ASSERT_NE(dispatcher->vkCmdPipelineBarrier2, nullptr);
    dispatcher->vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(commandBuffer),
                                      reinterpret_cast<const VkDependencyInfo *>(&dependencyInfo));
}

void insertAppBufferHandoffBarrier(const vk::raii::Device &device, vk::CommandBuffer commandBuffer, vk::Buffer buffer,
                                   vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) {
    vk::BufferMemoryBarrier2 bufferBarrier;
    bufferBarrier.srcStageMask = srcStage;
    bufferBarrier.srcAccessMask = srcAccess;
    bufferBarrier.dstStageMask = dstStage;
    bufferBarrier.dstAccessMask = dstAccess;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    vk::DependencyInfo dependencyInfo;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &bufferBarrier;

    const auto *const dispatcher = device.getDispatcher();
    ASSERT_NE(dispatcher->vkCmdPipelineBarrier2, nullptr);
    dispatcher->vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(commandBuffer),
                                      reinterpret_cast<const VkDependencyInfo *>(&dependencyInfo));
}

Workload makeOutputAliasedToIntermediateBufferWorkload() {
    constexpr uint32_t aliasGroup = 11;
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    WorkloadBuilder builder;
    const auto lhs = builder.addResource("lhs", makeBufferRequirements(bufferSize),
                                         WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto rhs = builder.addResource("rhs", makeBufferRequirements(bufferSize),
                                         WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto zero = builder.addResource("zero", makeBufferRequirements(bufferSize),
                                          WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto finalOutput = builder.addResource("final_output", makeBufferRequirements(bufferSize),
                                                 WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write));
    const auto aliasedOutput =
        builder.addResource("aliased_output", makeBufferRequirements(bufferSize),
                            WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write), aliasGroup);
    const auto intermediate = builder.addResource("intermediate", makeBufferRequirements(bufferSize),
                                                  Resource::Role::Intermediate, aliasGroup);
    const auto module = builder.addModule(makeSpirvModule(assembleAddInt32BuffersSpirv()), "add_int32_buffers", "main");

    const auto first = builder.addExecutable("write_intermediate_alias", ExecutableKind::Compute, module);
    builder.setDispatchShape(first, DispatchShape{10, 1, 1});
    builder.addDescriptorBinding(first, lhs, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(first, rhs, 0, 1, ResourceAccess::Read);
    builder.addDescriptorBinding(first, intermediate, 1, 2, ResourceAccess::Write);

    const auto second = builder.addExecutable("read_output_alias", ExecutableKind::Compute, module);
    builder.setDispatchShape(second, DispatchShape{10, 1, 1});
    builder.addDescriptorBinding(second, aliasedOutput, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(second, zero, 0, 1, ResourceAccess::Read);
    builder.addDescriptorBinding(second, finalOutput, 1, 2, ResourceAccess::Write);

    return builder.finish();
}

Workload makeOutputBufferAliasedToIntermediateTensorWorkload() {
    constexpr uint32_t aliasGroup = 17;
    constexpr vk::DeviceSize aliasedBufferSize = 64 * sizeof(int32_t);
    constexpr vk::DeviceSize outputBufferSize = 10 * sizeof(int32_t);

    WorkloadBuilder builder;
    const auto tensorInput =
        builder.addResource("tensor_input", makeTensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16}),
                            WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto aliasedOutput =
        builder.addResource("aliased_output", makeBufferRequirements(aliasedBufferSize),
                            WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write), aliasGroup);
    const auto zeroInput = builder.addResource("zero", makeBufferRequirements(outputBufferSize),
                                               WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto finalOutput = builder.addResource("final_output", makeBufferRequirements(outputBufferSize),
                                                 WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write));
    const auto intermediate =
        builder.addResource("intermediate_tensor", makeTensorRequirements(vk::Format::eR8Sint, {1, 4, 4, 16}),
                            Resource::Role::Intermediate, aliasGroup);

    const auto graphModule =
        builder.addModule(makeSpirvModule(assembleMaxpool8x8To4x4Spirv("composed_mixed_alias_graph", {0, 0, 0, 1})),
                          "maxpool_8x8_to_4x4", "main");
    const auto computeModule =
        builder.addModule(makeSpirvModule(assembleAddInt32BuffersSpirv()), "add_int32_buffers", "main");

    const auto graph = builder.addExecutable("write_intermediate_tensor_alias", ExecutableKind::Graph, graphModule);
    builder.addDescriptorBinding(graph, tensorInput, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(graph, intermediate, 0, 1, ResourceAccess::Write);

    const auto compute =
        builder.addExecutable("read_bound_output_buffer_alias", ExecutableKind::Compute, computeModule);
    builder.setDispatchShape(compute, DispatchShape{10, 1, 1});
    builder.addDescriptorBinding(compute, aliasedOutput, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(compute, zeroInput, 0, 1, ResourceAccess::Read);
    builder.addDescriptorBinding(compute, finalOutput, 1, 2, ResourceAccess::Write);

    return builder.finish();
}

Workload makeOutputTensorAliasedToIntermediateBufferWorkload() {
    constexpr uint32_t aliasGroup = 23;
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    WorkloadBuilder builder;
    const auto lhs = builder.addResource("lhs", makeBufferRequirements(bufferSize),
                                         WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto rhs = builder.addResource("rhs", makeBufferRequirements(bufferSize),
                                         WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto aliasedOutput = builder.addResource(
        "aliased_output_tensor", makeTensorRequirements(vk::Format::eR8Sint, {static_cast<int64_t>(bufferSize)}),
        WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write), aliasGroup);
    const auto intermediate = builder.addResource("intermediate_buffer", makeBufferRequirements(bufferSize),
                                                  Resource::Role::Intermediate, aliasGroup);
    const auto module = builder.addModule(makeSpirvModule(assembleAddInt32BuffersSpirv()), "add_int32_buffers", "main");

    const auto executable =
        builder.addExecutable("write_buffer_alias_of_output_tensor", ExecutableKind::Compute, module);
    builder.setDispatchShape(executable, DispatchShape{10, 1, 1});
    builder.addDescriptorBinding(executable, lhs, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(executable, rhs, 0, 1, ResourceAccess::Read);
    builder.addDescriptorBinding(executable, intermediate, 1, 2, ResourceAccess::Write);
    (void)aliasedOutput;

    return builder.finish();
}

Workload makeOutputTensorAliasedToIntermediateTensorWorkload() {
    constexpr uint32_t aliasGroup = 31;

    WorkloadBuilder builder;
    const auto input = builder.addResource("tensor_input", makeTensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16}),
                                           WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto aliasedOutput =
        builder.addResource("aliased_output_tensor", makeTensorRequirements(vk::Format::eR8Sint, {1, 4, 4, 16}),
                            WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write), aliasGroup);
    const auto intermediate =
        builder.addResource("intermediate_tensor", makeTensorRequirements(vk::Format::eR8Sint, {1, 4, 4, 16}),
                            Resource::Role::Intermediate, aliasGroup);
    const auto module =
        builder.addModule(makeSpirvModule(assembleMaxpool8x8To4x4Spirv("composed_tensor_alias", {0, 0, 0, 1})),
                          "maxpool_8x8_to_4x4", "main");

    const auto executable = builder.addExecutable("write_tensor_alias_of_output_tensor", ExecutableKind::Graph, module);
    builder.addDescriptorBinding(executable, input, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(executable, intermediate, 0, 1, ResourceAccess::Write);
    (void)aliasedOutput;

    return builder.finish();
}

Workload makeOutputImageAliasedToIntermediateImageWorkload() {
    constexpr uint32_t aliasGroup = 37;
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    WorkloadBuilder builder;
    const auto lhs = builder.addResource("lhs", makeBufferRequirements(bufferSize),
                                         WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto rhs = builder.addResource("rhs", makeBufferRequirements(bufferSize),
                                         WorkloadBuilder::publicRoleForAccess(ResourceAccess::Read));
    const auto output = builder.addResource("output", makeBufferRequirements(bufferSize),
                                            WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write));
    const auto aliasedOutput =
        builder.addResource("aliased_output_image", makeStorageImageRequirements(),
                            WorkloadBuilder::publicRoleForAccess(ResourceAccess::Write), aliasGroup);
    const auto intermediate = builder.addResource("intermediate_image", makeStorageImageRequirements(),
                                                  Resource::Role::Intermediate, aliasGroup);
    const auto module = builder.addModule(makeSpirvModule(assembleAddInt32BuffersSpirv()), "add_int32_buffers", "main");

    const auto executable = builder.addExecutable("write_image_alias_metadata", ExecutableKind::Compute, module);
    builder.setDispatchShape(executable, DispatchShape{10, 1, 1});
    builder.addDescriptorBinding(executable, lhs, 0, 0, ResourceAccess::Read);
    builder.addDescriptorBinding(executable, rhs, 0, 1, ResourceAccess::Read);
    builder.addDescriptorBinding(executable, output, 1, 2, ResourceAccess::Write);
    builder.addDescriptorBinding(executable, intermediate, 1, 3, ResourceAccess::Write);
    (void)aliasedOutput;

    return builder.finish();
}

Workload makeAliasWorkload(AliasScenario scenario) {
    switch (scenario) {
    case AliasScenario::OutputBufferAliasedToIntermediateBuffer:
        return makeOutputAliasedToIntermediateBufferWorkload();
    case AliasScenario::OutputBufferAliasedToIntermediateTensor:
        return makeOutputBufferAliasedToIntermediateTensorWorkload();
    case AliasScenario::OutputTensorAliasedToIntermediateBuffer:
        return makeOutputTensorAliasedToIntermediateBufferWorkload();
    case AliasScenario::OutputTensorAliasedToIntermediateTensor:
        return makeOutputTensorAliasedToIntermediateTensorWorkload();
    case AliasScenario::OutputImageAliasedToIntermediateImage:
        return makeOutputImageAliasedToIntermediateImageWorkload();
    }
    throw std::logic_error("Unhandled alias scenario");
}

} // namespace

class ComposedAliasExecutionTest : public AliasExecutionTestBase, public testing::WithParamInterface<AliasCase> {};

TEST_P(ComposedAliasExecutionTest, Run) {
    auto workload = makeAliasWorkload(GetParam().scenario);
    runAliasCase(workload, GetParam(), true);
}

TEST_P(ComposedAliasExecutionTest, PrepareRequiresBoundMemoryInfo) {
    auto workload = makeAliasWorkload(GetParam().scenario);
    runAliasCase(workload, GetParam(), false);
}

INSTANTIATE_TEST_SUITE_P(AliasCases, ComposedAliasExecutionTest,
                         testing::Values(AliasCase{"OutputBufferAliasedToIntermediateBuffer",
                                                   AliasScenario::OutputBufferAliasedToIntermediateBuffer, 2, 5},
                                         AliasCase{"OutputBufferAliasedToIntermediateTensor",
                                                   AliasScenario::OutputBufferAliasedToIntermediateTensor, 2, 4},
                                         AliasCase{"OutputTensorAliasedToIntermediateBuffer",
                                                   AliasScenario::OutputTensorAliasedToIntermediateBuffer, 1, 3},
                                         AliasCase{"OutputTensorAliasedToIntermediateTensor",
                                                   AliasScenario::OutputTensorAliasedToIntermediateTensor, 1, 2},
                                         AliasCase{"OutputImageAliasedToIntermediateImage",
                                                   AliasScenario::OutputImageAliasedToIntermediateImage, 1, 4}),
                         aliasCaseName);

TEST_F(ComposedSessionExecutionTest, RecordTwoSagWorkloadsWithAppTensorHandoff) {
    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto firstSag = Workload::fromDataGraph(makeMaxpoolDescription());
    auto secondSag = Workload::fromDataGraph(makeMaxpool8x8To4x4Description());
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor intermediateTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto input = makeMaxpoolInput(inputTensor.shape, 7);
    inputTensor.write(input);
    intermediateTensor.fill(0, intermediateTensor.numElements());
    outputTensor.fill(0, outputTensor.numElements());

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session firstSession(context, firstSag);
    Session secondSession(context, secondSag);
    firstSession.configure();
    secondSession.configure();

    auto firstBindings = firstSession.createBindingSet();
    firstBindings.bindTensor(firstSag.resource(0), TensorBindingInfo{*inputTensor.tensor});
    firstBindings.bindTensor(firstSag.resource(1), TensorBindingInfo{*intermediateTensor.tensor});

    auto secondBindings = secondSession.createBindingSet();
    secondBindings.bindTensor(secondSag.resource(0), TensorBindingInfo{*intermediateTensor.tensor});
    secondBindings.bindTensor(secondSag.resource(1), TensorBindingInfo{*outputTensor.tensor});

    auto firstExecution = firstSession.prepare(firstBindings);
    auto secondExecution = secondSession.prepare(secondBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        firstExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(
            device, commandBuffer, *intermediateTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
        secondExecution.record(commandBuffer);
    });

    /***************************************************************************
     * Verification
     **************************************************************************/

    const auto firstExpected = expectedMaxpool(input, inputTensor.shape);
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(firstExpected, {1, 8, 8, 16}));
}

TEST_F(ComposedSessionExecutionTest, RecordTwoSagWorkloadsRepeatedDifferentInput) {
    /***************************************************************************
     * Workloads, context, and reusable execution state
     **************************************************************************/

    auto firstSag = Workload::fromDataGraph(makeMaxpoolDescription());
    auto secondSag = Workload::fromDataGraph(makeMaxpool8x8To4x4Description());
    auto context = wrappedContext();

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor intermediateTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    Session firstSession(context, firstSag);
    Session secondSession(context, secondSag);
    firstSession.configure();
    secondSession.configure();

    auto firstBindings = firstSession.createBindingSet();
    firstBindings.bindTensor(firstSag.resource(0), TensorBindingInfo{*inputTensor.tensor});
    firstBindings.bindTensor(firstSag.resource(1), TensorBindingInfo{*intermediateTensor.tensor});

    auto secondBindings = secondSession.createBindingSet();
    secondBindings.bindTensor(secondSag.resource(0), TensorBindingInfo{*intermediateTensor.tensor});
    secondBindings.bindTensor(secondSag.resource(1), TensorBindingInfo{*outputTensor.tensor});

    auto firstExecution = firstSession.prepare(firstBindings);
    auto secondExecution = secondSession.prepare(secondBindings);

    for (const auto seed : {7U, 23U, 47U}) {
        /***************************************************************************
         * Per-iteration data
         **************************************************************************/

        const auto input = makeMaxpoolInput(inputTensor.shape, seed);
        inputTensor.write(input);
        intermediateTensor.fill(0, intermediateTensor.numElements());
        outputTensor.fill(0, outputTensor.numElements());

        recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
            firstExecution.record(commandBuffer);
            insertAppTensorHandoffBarrier(
                device, commandBuffer, *intermediateTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
                vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eDataGraphARM,
                vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
            secondExecution.record(commandBuffer);
        });

        const auto firstExpected = expectedMaxpool(input, inputTensor.shape);
        EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(firstExpected, {1, 8, 8, 16}));
    }
}

TEST_F(ComposedSessionExecutionTest, RecordTwoSacWorkloadsWithAppBufferHandoff) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto firstSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto secondSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Buffer lhsBuffer(physicalDevice, device, bufferSize);
    const Buffer rhsBuffer(physicalDevice, device, bufferSize);
    const Buffer zeroBuffer(physicalDevice, device, bufferSize);
    const Buffer intermediateBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> lhs = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
    const std::vector<int32_t> rhs = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    const std::vector<int32_t> zeros(elements, 0);
    lhsBuffer.write(lhs);
    rhsBuffer.write(rhs);
    zeroBuffer.write(zeros);
    intermediateBuffer.write(zeros);
    outputBuffer.write(zeros);

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session firstSession(context, firstSac);
    Session secondSession(context, secondSac);
    firstSession.configure();
    secondSession.configure();

    auto firstBindings = firstSession.createBindingSet();
    firstBindings.bindBuffer(firstSac.resource(0), BufferBindingInfo{*lhsBuffer.buffer});
    firstBindings.bindBuffer(firstSac.resource(1), BufferBindingInfo{*rhsBuffer.buffer});
    firstBindings.bindBuffer(firstSac.resource(2), BufferBindingInfo{*intermediateBuffer.buffer});

    auto secondBindings = secondSession.createBindingSet();
    secondBindings.bindBuffer(secondSac.resource(0), BufferBindingInfo{*intermediateBuffer.buffer});
    secondBindings.bindBuffer(secondSac.resource(1), BufferBindingInfo{*zeroBuffer.buffer});
    secondBindings.bindBuffer(secondSac.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto firstExecution = firstSession.prepare(firstBindings);
    auto secondExecution = secondSession.prepare(secondBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        firstExecution.record(commandBuffer);
        insertAppBufferHandoffBarrier(device, commandBuffer, *intermediateBuffer.buffer,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        secondExecution.record(commandBuffer);
    });

    EXPECT_EQ(outputBuffer.read(elements), addVectors(lhs, rhs));
}

TEST_F(ComposedSessionExecutionTest, RunSacFanOutDagFromOneSacOutputToTwoSacReaders) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto sourceSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto firstReaderSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto secondReaderSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Buffer sourceSacLhsBuffer(physicalDevice, device, bufferSize);
    const Buffer sourceSacRhsBuffer(physicalDevice, device, bufferSize);
    const Buffer sourceSacOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer firstReaderSacRhsBuffer(physicalDevice, device, bufferSize);
    const Buffer firstOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondReaderSacRhsBuffer(physicalDevice, device, bufferSize);
    const Buffer secondOutputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> sourceSacLhs = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
    const std::vector<int32_t> sourceSacRhs = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    const std::vector<int32_t> firstReaderSacRhs = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
    const std::vector<int32_t> secondReaderSacRhs = {-7, -11, -13, -17, -19, -23, -29, -31, -37, -41};
    const std::vector<int32_t> zeros(elements, 0);
    sourceSacLhsBuffer.write(sourceSacLhs);
    sourceSacRhsBuffer.write(sourceSacRhs);
    sourceSacOutputBuffer.write(zeros);
    firstReaderSacRhsBuffer.write(firstReaderSacRhs);
    firstOutputBuffer.write(zeros);
    secondReaderSacRhsBuffer.write(secondReaderSacRhs);
    secondOutputBuffer.write(zeros);

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session sourceSession(context, sourceSac);
    Session firstReaderSession(context, firstReaderSac);
    Session secondReaderSession(context, secondReaderSac);
    sourceSession.configure();
    firstReaderSession.configure();
    secondReaderSession.configure();

    auto sourceBindings = sourceSession.createBindingSet();
    sourceBindings.bindBuffer(sourceSac.resource(0), BufferBindingInfo{*sourceSacLhsBuffer.buffer});
    sourceBindings.bindBuffer(sourceSac.resource(1), BufferBindingInfo{*sourceSacRhsBuffer.buffer});
    sourceBindings.bindBuffer(sourceSac.resource(2), BufferBindingInfo{*sourceSacOutputBuffer.buffer});

    auto firstReaderBindings = firstReaderSession.createBindingSet();
    firstReaderBindings.bindBuffer(firstReaderSac.resource(0), BufferBindingInfo{*sourceSacOutputBuffer.buffer});
    firstReaderBindings.bindBuffer(firstReaderSac.resource(1), BufferBindingInfo{*firstReaderSacRhsBuffer.buffer});
    firstReaderBindings.bindBuffer(firstReaderSac.resource(2), BufferBindingInfo{*firstOutputBuffer.buffer});

    auto secondReaderBindings = secondReaderSession.createBindingSet();
    secondReaderBindings.bindBuffer(secondReaderSac.resource(0), BufferBindingInfo{*sourceSacOutputBuffer.buffer});
    secondReaderBindings.bindBuffer(secondReaderSac.resource(1), BufferBindingInfo{*secondReaderSacRhsBuffer.buffer});
    secondReaderBindings.bindBuffer(secondReaderSac.resource(2), BufferBindingInfo{*secondOutputBuffer.buffer});

    auto sourceExecution = sourceSession.prepare(sourceBindings);
    auto firstReaderExecution = firstReaderSession.prepare(firstReaderBindings);
    auto secondReaderExecution = secondReaderSession.prepare(secondReaderBindings);

    /***************************************************************************
     * Application-side DAG sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        sourceExecution.record(commandBuffer);
        insertAppBufferHandoffBarrier(device, commandBuffer, *sourceSacOutputBuffer.buffer,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        firstReaderExecution.record(commandBuffer);
        secondReaderExecution.record(commandBuffer);
    });

    const auto sourceSacOutput = addVectors(sourceSacLhs, sourceSacRhs);
    EXPECT_EQ(firstOutputBuffer.read(elements), addVectors(sourceSacOutput, firstReaderSacRhs));
    EXPECT_EQ(secondOutputBuffer.read(elements), addVectors(sourceSacOutput, secondReaderSacRhs));
}

TEST_F(ComposedSessionExecutionTest, RunSacFanInDagFromTwoSacOutputsToOneSacReader) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto firstSourceSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto secondSourceSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto readerSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Buffer firstSourceSacLhsBuffer(physicalDevice, device, bufferSize);
    const Buffer firstSourceSacRhsBuffer(physicalDevice, device, bufferSize);
    const Buffer firstSourceSacOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondSourceSacLhsBuffer(physicalDevice, device, bufferSize);
    const Buffer secondSourceSacRhsBuffer(physicalDevice, device, bufferSize);
    const Buffer secondSourceSacOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstSourceSacLhs = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
    const std::vector<int32_t> firstSourceSacRhs = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    const std::vector<int32_t> secondSourceSacLhs = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
    const std::vector<int32_t> secondSourceSacRhs = {-7, -11, -13, -17, -19, -23, -29, -31, -37, -41};
    firstSourceSacLhsBuffer.write(firstSourceSacLhs);
    firstSourceSacRhsBuffer.write(firstSourceSacRhs);
    firstSourceSacOutputBuffer.write(std::vector<int32_t>(elements, 0));
    secondSourceSacLhsBuffer.write(secondSourceSacLhs);
    secondSourceSacRhsBuffer.write(secondSourceSacRhs);
    secondSourceSacOutputBuffer.write(std::vector<int32_t>(elements, 0));
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session firstSourceSession(context, firstSourceSac);
    Session secondSourceSession(context, secondSourceSac);
    Session readerSession(context, readerSac);
    firstSourceSession.configure();
    secondSourceSession.configure();
    readerSession.configure();

    auto firstSourceBindings = firstSourceSession.createBindingSet();
    firstSourceBindings.bindBuffer(firstSourceSac.resource(0), BufferBindingInfo{*firstSourceSacLhsBuffer.buffer});
    firstSourceBindings.bindBuffer(firstSourceSac.resource(1), BufferBindingInfo{*firstSourceSacRhsBuffer.buffer});
    firstSourceBindings.bindBuffer(firstSourceSac.resource(2), BufferBindingInfo{*firstSourceSacOutputBuffer.buffer});

    auto secondSourceBindings = secondSourceSession.createBindingSet();
    secondSourceBindings.bindBuffer(secondSourceSac.resource(0), BufferBindingInfo{*secondSourceSacLhsBuffer.buffer});
    secondSourceBindings.bindBuffer(secondSourceSac.resource(1), BufferBindingInfo{*secondSourceSacRhsBuffer.buffer});
    secondSourceBindings.bindBuffer(secondSourceSac.resource(2),
                                    BufferBindingInfo{*secondSourceSacOutputBuffer.buffer});

    auto readerBindings = readerSession.createBindingSet();
    readerBindings.bindBuffer(readerSac.resource(0), BufferBindingInfo{*firstSourceSacOutputBuffer.buffer});
    readerBindings.bindBuffer(readerSac.resource(1), BufferBindingInfo{*secondSourceSacOutputBuffer.buffer});
    readerBindings.bindBuffer(readerSac.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto firstSourceExecution = firstSourceSession.prepare(firstSourceBindings);
    auto secondSourceExecution = secondSourceSession.prepare(secondSourceBindings);
    auto readerExecution = readerSession.prepare(readerBindings);

    /***************************************************************************
     * Application-side DAG sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        firstSourceExecution.record(commandBuffer);
        secondSourceExecution.record(commandBuffer);
        insertAppBufferHandoffBarrier(device, commandBuffer, *firstSourceSacOutputBuffer.buffer,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        insertAppBufferHandoffBarrier(device, commandBuffer, *secondSourceSacOutputBuffer.buffer,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        readerExecution.record(commandBuffer);
    });

    const auto firstSourceSacOutput = addVectors(firstSourceSacLhs, firstSourceSacRhs);
    const auto secondSourceSacOutput = addVectors(secondSourceSacLhs, secondSourceSacRhs);
    EXPECT_EQ(outputBuffer.read(elements), addVectors(firstSourceSacOutput, secondSourceSacOutput));
}

TEST_F(ComposedSessionExecutionTest, RecordSacVgfSacSequenceWithoutLibraryComposition) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled";
    }

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto sacAdd = Workload::fromComputeShader(makeTensorAddDescription({1, 16, 16, 16}));
    const auto vgfBytes = makeMaxpool16x16To8x8Vgf();
    auto vgfMaxpool = Workload::fromVGF(vgfBytes.data(), vgfBytes.size());
    auto sacSubtract = Workload::fromComputeShader(makeTensorSubtractDescription({1, 8, 8, 16}));
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor secondInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor addOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor graphOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor subtractInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});

    const auto firstInput = makeSmallTensorInput(firstInputTensor.shape, 5);
    const auto secondInput = makeSmallTensorInput(secondInputTensor.shape, 13);
    const auto subtractInput = makeSmallTensorInput(subtractInputTensor.shape, 17);
    firstInputTensor.write(firstInput);
    secondInputTensor.write(secondInput);
    addOutputTensor.fill(0, addOutputTensor.numElements());
    graphOutputTensor.fill(0, graphOutputTensor.numElements());
    subtractInputTensor.write(subtractInput);
    outputTensor.fill(0, outputTensor.numElements());

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session addSession(context, sacAdd);
    Session graphSession(context, vgfMaxpool);
    Session subtractSession(context, sacSubtract);
    addSession.configure();
    graphSession.configure();
    subtractSession.configure();

    auto addBindings = addSession.createBindingSet();
    addBindings.bindTensor(sacAdd.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    addBindings.bindTensor(sacAdd.resource(1), TensorBindingInfo{*secondInputTensor.tensor});
    addBindings.bindTensor(sacAdd.resource(2), TensorBindingInfo{*addOutputTensor.tensor});

    auto graphBindings = graphSession.createBindingSet();
    graphBindings.bindTensor(vgfMaxpool.resource(0), TensorBindingInfo{*addOutputTensor.tensor});
    graphBindings.bindTensor(vgfMaxpool.resource(1), TensorBindingInfo{*graphOutputTensor.tensor});

    auto subtractBindings = subtractSession.createBindingSet();
    subtractBindings.bindTensor(sacSubtract.resource(0), TensorBindingInfo{*graphOutputTensor.tensor});
    subtractBindings.bindTensor(sacSubtract.resource(1), TensorBindingInfo{*subtractInputTensor.tensor});
    subtractBindings.bindTensor(sacSubtract.resource(2), TensorBindingInfo{*outputTensor.tensor});

    auto addExecution = addSession.prepare(addBindings);
    auto graphExecution = graphSession.prepare(graphBindings);
    auto subtractExecution = subtractSession.prepare(subtractBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        addExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(device, commandBuffer, *addOutputTensor.tensor,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eDataGraphARM,
                                      vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
        graphExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(
            device, commandBuffer, *graphOutputTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        subtractExecution.record(commandBuffer);
    });

    const auto addedInput = addTensors(firstInput, secondInput);
    const auto graphOutput = expectedMaxpool(addedInput, firstInputTensor.shape);
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), subtractTensors(graphOutput, subtractInput));
}

TEST_F(ComposedSessionExecutionTest, RecordSacMultiSegmentVgfSacSequenceWithoutLibraryComposition) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled";
    }

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto sacAdd = Workload::fromComputeShader(makeTensorAddDescription({1, 16, 16, 16}));
    const auto vgfBytes = makeTwoSegmentMaxpoolVgf();
    auto vgfMaxpool = Workload::fromVGF(vgfBytes.data(), vgfBytes.size());
    auto sacSubtract = Workload::fromComputeShader(makeTensorSubtractDescription({1, 4, 4, 16}));
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor secondInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor addOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor graphOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});
    const Tensor subtractInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto firstInput = makeSmallTensorInput(firstInputTensor.shape, 19);
    const auto secondInput = makeSmallTensorInput(secondInputTensor.shape, 23);
    const auto subtractInput = makeSmallTensorInput(subtractInputTensor.shape, 29);
    firstInputTensor.write(firstInput);
    secondInputTensor.write(secondInput);
    addOutputTensor.fill(0, addOutputTensor.numElements());
    graphOutputTensor.fill(0, graphOutputTensor.numElements());
    subtractInputTensor.write(subtractInput);
    outputTensor.fill(0, outputTensor.numElements());

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session addSession(context, sacAdd);
    Session graphSession(context, vgfMaxpool);
    Session subtractSession(context, sacSubtract);
    addSession.configure();
    graphSession.configure();
    subtractSession.configure();

    auto addBindings = addSession.createBindingSet();
    addBindings.bindTensor(sacAdd.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    addBindings.bindTensor(sacAdd.resource(1), TensorBindingInfo{*secondInputTensor.tensor});
    addBindings.bindTensor(sacAdd.resource(2), TensorBindingInfo{*addOutputTensor.tensor});

    auto graphBindings = graphSession.createBindingSet();
    graphBindings.bindTensor(vgfMaxpool.resource(0), TensorBindingInfo{*addOutputTensor.tensor});
    graphBindings.bindTensor(vgfMaxpool.resource(1), TensorBindingInfo{*graphOutputTensor.tensor});

    auto subtractBindings = subtractSession.createBindingSet();
    subtractBindings.bindTensor(sacSubtract.resource(0), TensorBindingInfo{*graphOutputTensor.tensor});
    subtractBindings.bindTensor(sacSubtract.resource(1), TensorBindingInfo{*subtractInputTensor.tensor});
    subtractBindings.bindTensor(sacSubtract.resource(2), TensorBindingInfo{*outputTensor.tensor});

    auto addExecution = addSession.prepare(addBindings);
    auto graphExecution = graphSession.prepare(graphBindings);
    auto subtractExecution = subtractSession.prepare(subtractBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        addExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(device, commandBuffer, *addOutputTensor.tensor,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eDataGraphARM,
                                      vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
        graphExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(
            device, commandBuffer, *graphOutputTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        subtractExecution.record(commandBuffer);
    });

    const auto addedInput = addTensors(firstInput, secondInput);
    const auto firstGraphOutput = expectedMaxpool(addedInput, firstInputTensor.shape);
    const auto secondGraphOutput = expectedMaxpool(firstGraphOutput, {1, 8, 8, 16});
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), subtractTensors(secondGraphOutput, subtractInput));
}

TEST_F(ComposedSessionExecutionTest, RecordSacVgfSagSacSequenceWithoutLibraryComposition) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled";
    }

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto sacAdd = Workload::fromComputeShader(makeTensorAddDescription({1, 16, 16, 16}));
    const auto vgfBytes = makeMaxpool16x16To8x8Vgf();
    auto vgfMaxpool = Workload::fromVGF(vgfBytes.data(), vgfBytes.size());
    auto sagMaxpool = Workload::fromDataGraph(makeMaxpool8x8To4x4Description());
    auto sacSubtract = Workload::fromComputeShader(makeTensorSubtractDescription({1, 4, 4, 16}));
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor secondInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor sacAddOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor vgfOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor sagOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});
    const Tensor subtractInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto firstInput = makeSmallTensorInput(firstInputTensor.shape, 31);
    const auto secondInput = makeSmallTensorInput(secondInputTensor.shape, 37);
    const auto subtractInput = makeSmallTensorInput(subtractInputTensor.shape, 41);
    firstInputTensor.write(firstInput);
    secondInputTensor.write(secondInput);
    sacAddOutputTensor.fill(0, sacAddOutputTensor.numElements());
    vgfOutputTensor.fill(0, vgfOutputTensor.numElements());
    sagOutputTensor.fill(0, sagOutputTensor.numElements());
    subtractInputTensor.write(subtractInput);
    outputTensor.fill(0, outputTensor.numElements());

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session addSession(context, sacAdd);
    Session vgfSession(context, vgfMaxpool);
    Session sagSession(context, sagMaxpool);
    Session subtractSession(context, sacSubtract);
    addSession.configure();
    vgfSession.configure();
    sagSession.configure();
    subtractSession.configure();

    auto addBindings = addSession.createBindingSet();
    addBindings.bindTensor(sacAdd.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    addBindings.bindTensor(sacAdd.resource(1), TensorBindingInfo{*secondInputTensor.tensor});
    addBindings.bindTensor(sacAdd.resource(2), TensorBindingInfo{*sacAddOutputTensor.tensor});

    auto vgfBindings = vgfSession.createBindingSet();
    vgfBindings.bindTensor(vgfMaxpool.resource(0), TensorBindingInfo{*sacAddOutputTensor.tensor});
    vgfBindings.bindTensor(vgfMaxpool.resource(1), TensorBindingInfo{*vgfOutputTensor.tensor});

    auto sagBindings = sagSession.createBindingSet();
    sagBindings.bindTensor(sagMaxpool.resource(0), TensorBindingInfo{*vgfOutputTensor.tensor});
    sagBindings.bindTensor(sagMaxpool.resource(1), TensorBindingInfo{*sagOutputTensor.tensor});

    auto subtractBindings = subtractSession.createBindingSet();
    subtractBindings.bindTensor(sacSubtract.resource(0), TensorBindingInfo{*sagOutputTensor.tensor});
    subtractBindings.bindTensor(sacSubtract.resource(1), TensorBindingInfo{*subtractInputTensor.tensor});
    subtractBindings.bindTensor(sacSubtract.resource(2), TensorBindingInfo{*outputTensor.tensor});

    auto addExecution = addSession.prepare(addBindings);
    auto vgfExecution = vgfSession.prepare(vgfBindings);
    auto sagExecution = sagSession.prepare(sagBindings);
    auto subtractExecution = subtractSession.prepare(subtractBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        addExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(device, commandBuffer, *sacAddOutputTensor.tensor,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eDataGraphARM,
                                      vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
        vgfExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(
            device, commandBuffer, *vgfOutputTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
        sagExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(
            device, commandBuffer, *sagOutputTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        subtractExecution.record(commandBuffer);
    });

    const auto addedInput = addTensors(firstInput, secondInput);
    const auto vgfOutput = expectedMaxpool(addedInput, firstInputTensor.shape);
    const auto sagOutput = expectedMaxpool(vgfOutput, {1, 8, 8, 16});
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), subtractTensors(sagOutput, subtractInput));
}

TEST_F(ComposedSessionExecutionTest, RecordSagVgfSequenceWithoutLibraryComposition) {
    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto sagMaxpool = Workload::fromDataGraph(makeMaxpoolDescription());
    const auto vgfBytes = makeMaxpool8x8To4x4Vgf();
    auto vgfMaxpool = Workload::fromVGF(vgfBytes.data(), vgfBytes.size());
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor intermediateTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto input = makeMaxpoolInput(inputTensor.shape, 31);
    inputTensor.write(input);
    intermediateTensor.fill(0, intermediateTensor.numElements());
    outputTensor.fill(0, outputTensor.numElements());

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session sagSession(context, sagMaxpool);
    Session vgfSession(context, vgfMaxpool);
    sagSession.configure();
    vgfSession.configure();

    auto sagBindings = sagSession.createBindingSet();
    sagBindings.bindTensor(sagMaxpool.resource(0), TensorBindingInfo{*inputTensor.tensor});
    sagBindings.bindTensor(sagMaxpool.resource(1), TensorBindingInfo{*intermediateTensor.tensor});

    auto vgfBindings = vgfSession.createBindingSet();
    vgfBindings.bindTensor(vgfMaxpool.resource(0), TensorBindingInfo{*intermediateTensor.tensor});
    vgfBindings.bindTensor(vgfMaxpool.resource(1), TensorBindingInfo{*outputTensor.tensor});

    auto sagExecution = sagSession.prepare(sagBindings);
    auto vgfExecution = vgfSession.prepare(vgfBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        sagExecution.record(commandBuffer);
        insertAppTensorHandoffBarrier(
            device, commandBuffer, *intermediateTensor.tensor, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphWriteARM, vk::PipelineStageFlagBits2::eDataGraphARM,
            vk::AccessFlagBits2::eDataGraphReadARM | vk::AccessFlagBits2::eDataGraphWriteARM);
        vgfExecution.record(commandBuffer);
    });

    const auto firstExpected = expectedMaxpool(input, inputTensor.shape);
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(firstExpected, {1, 8, 8, 16}));
}

TEST_F(ComposedSessionExecutionTest, RecordSacVgfSacBufferSequenceWithoutLibraryComposition) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    /***************************************************************************
     * Workloads and context
     **************************************************************************/

    auto firstSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    const auto vgfBytes = makeAddInt32BuffersVgf();
    auto vgfAdd = Workload::fromVGF(vgfBytes.data(), vgfBytes.size());
    auto secondSac = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    /***************************************************************************
     * Application-owned resources
     **************************************************************************/

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer firstSacOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer vgfSecondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer vgfOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondSacSecondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
    const std::vector<int32_t> secondInput = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    const std::vector<int32_t> vgfSecondInput = {-2, -3, -5, -7, -11, -13, -17, -19, -23, -29};
    const std::vector<int32_t> secondSacSecondInput = {30, 29, 28, 27, 26, 25, 24, 23, 22, 21};
    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    firstSacOutputBuffer.write(std::vector<int32_t>(elements, 0));
    vgfSecondInputBuffer.write(vgfSecondInput);
    vgfOutputBuffer.write(std::vector<int32_t>(elements, 0));
    secondSacSecondInputBuffer.write(secondSacSecondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    /***************************************************************************
     * Per-workload sessions and bindings
     **************************************************************************/

    Session firstSacSession(context, firstSac);
    Session vgfSession(context, vgfAdd);
    Session secondSacSession(context, secondSac);
    firstSacSession.configure();
    vgfSession.configure();
    secondSacSession.configure();

    auto firstSacBindings = firstSacSession.createBindingSet();
    firstSacBindings.bindBuffer(firstSac.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    firstSacBindings.bindBuffer(firstSac.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    firstSacBindings.bindBuffer(firstSac.resource(2), BufferBindingInfo{*firstSacOutputBuffer.buffer});

    auto vgfBindings = vgfSession.createBindingSet();
    vgfBindings.bindBuffer(vgfAdd.resource(0), BufferBindingInfo{*firstSacOutputBuffer.buffer});
    vgfBindings.bindBuffer(vgfAdd.resource(1), BufferBindingInfo{*vgfSecondInputBuffer.buffer});
    vgfBindings.bindBuffer(vgfAdd.resource(2), BufferBindingInfo{*vgfOutputBuffer.buffer});

    auto secondSacBindings = secondSacSession.createBindingSet();
    secondSacBindings.bindBuffer(secondSac.resource(0), BufferBindingInfo{*vgfOutputBuffer.buffer});
    secondSacBindings.bindBuffer(secondSac.resource(1), BufferBindingInfo{*secondSacSecondInputBuffer.buffer});
    secondSacBindings.bindBuffer(secondSac.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto firstSacExecution = firstSacSession.prepare(firstSacBindings);
    auto vgfExecution = vgfSession.prepare(vgfBindings);
    auto secondSacExecution = secondSacSession.prepare(secondSacBindings);

    /***************************************************************************
     * Application-side sequencing
     **************************************************************************/

    recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
        firstSacExecution.record(commandBuffer);
        insertAppBufferHandoffBarrier(device, commandBuffer, *firstSacOutputBuffer.buffer,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        vgfExecution.record(commandBuffer);
        insertAppBufferHandoffBarrier(device, commandBuffer, *vgfOutputBuffer.buffer,
                                      vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                      vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        secondSacExecution.record(commandBuffer);
    });

    const auto firstOutput = addVectors(firstInput, secondInput);
    const auto vgfOutput = addVectors(firstOutput, vgfSecondInput);
    EXPECT_EQ(outputBuffer.read(elements), addVectors(vgfOutput, secondSacSecondInput));
}
