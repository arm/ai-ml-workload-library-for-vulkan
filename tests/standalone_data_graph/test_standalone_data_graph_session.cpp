/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_standalone_data_graph_utils.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

class StandaloneDataGraphSessionExecutionTest : public RuntimeSessionExecutionTest {};

BoundMemoryInfo boundMemoryInfo(const Tensor &tensor) { return {*tensor.memory, 0, tensor.memorySize}; }

std::vector<int8_t> makeSparse2To4Weights(std::size_t elements) {
    if (elements % 4 != 0) {
        throw std::runtime_error("2:4 sparse test weights must be a multiple of four");
    }

    std::vector<int8_t> weights(elements, 0);
    for (std::size_t offset = 0; offset < elements; offset += 4) {
        weights[offset] = static_cast<int8_t>(((offset / 4) % 31) + 1);
        weights[offset + 2] = static_cast<int8_t>((((offset / 4) * 3) % 29) + 1);
    }
    return weights;
}

} // namespace

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWorkload) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 11);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputShape));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWorkloadWithPipelineMetadata) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto description = makeMaxpoolDescription();
    description.pipeline.identifier = "maxpool-disable-optimization";
    description.pipeline.flags = vk::PipelineCreateFlagBits2::eDisableOptimization;
    auto workload = Workload::fromDataGraph(std::move(description));
    auto context = wrappedContext();

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 13);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputShape));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWorkloadWithPipelineSpecializationConstants) {
    auto context = wrappedContext();

    auto runArshift = [&](bool round) {
        const std::vector<int64_t> tensorShape = {4};
        auto description = makeArshiftSpecBoolDescription(round);
        description.pipeline.identifier = "arshift-disable-optimization";
        description.pipeline.flags = vk::PipelineCreateFlagBits2::eDisableOptimization;
        auto workload = Workload::fromDataGraph(std::move(description));

        const Tensor inputTensor(physicalDevice, device, vk::Format::eR16Uint, tensorShape);
        const Tensor shiftTensor(physicalDevice, device, vk::Format::eR16Uint, tensorShape);
        const Tensor outputTensor(physicalDevice, device, vk::Format::eR16Uint, tensorShape);

        const std::vector<uint16_t> input = {7, 9, 10, 15};
        const std::vector<uint16_t> shift = {1, 1, 2, 2};
        writeMappedMemory(device, boundMemoryInfo(inputTensor), input);
        writeMappedMemory(device, boundMemoryInfo(shiftTensor), shift);
        writeMappedMemory(device, boundMemoryInfo(outputTensor), std::vector<uint16_t>(input.size(), 0));

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        bindings.bindTensor(workload.resource(1), TensorBindingInfo{*shiftTensor.tensor});
        bindings.bindTensor(workload.resource(2), TensorBindingInfo{*outputTensor.tensor});

        auto execution = session.prepare(bindings);
        execution.run();

        return readMappedMemory<uint16_t>(device, boundMemoryInfo(outputTensor), input.size());
    };

    EXPECT_EQ(runArshift(false), (std::vector<uint16_t>{3, 4, 2, 3}));
    EXPECT_EQ(runArshift(true), (std::vector<uint16_t>{4, 5, 3, 4}));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWorkloadWithRuntimeCreatedContextAndTensors) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = Context::create();
    const auto contextView = context.contextView();

    auto inputTensor = context.createTensor(workload.resource(0));
    auto outputTensor = context.createTensor(workload.resource(1));

    const auto input = makeMaxpoolInput(inputShape, 19);
    writeMappedMemory(contextView.device, inputTensor.memory(), input);
    writeMappedMemory(contextView.device, outputTensor.memory(),
                      std::vector<int8_t>(Tensor::numElements(outputShape), 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{inputTensor.handle(), inputTensor.memory()});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{outputTensor.handle(), outputTensor.memory()});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(readMappedMemory<int8_t>(contextView.device, outputTensor.memory(), Tensor::numElements(outputShape)),
              expectedMaxpool(input, inputShape));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, MovesRuntimeOwnedTensorAllocations) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();

    auto tensor = context.createTensor(workload.resource(0));
    auto movedTensor(std::move(tensor));
    auto assignedTensor = context.createTensor(workload.resource(1));
    EXPECT_NE(assignedTensor.handle(), nullptr);
    assignedTensor = std::move(movedTensor);

    EXPECT_NE(assignedTensor.handle(), nullptr);
    EXPECT_NE(assignedTensor.memory().memory, nullptr);
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunAddConstantInputWithRuntimeCreatedContextAndTensors) {
    const std::vector<int64_t> tensorShape = {1, 3, 2, 1};

    const std::vector<float> constant(Tensor::numElements(tensorShape), 0.5F);
    auto workload = Workload::fromDataGraph(makeAddF32ConstantDescription(constant));
    auto context = Context::create();
    const auto contextView = context.contextView();

    auto inputTensor = context.createTensor(workload.resource(0));
    auto outputTensor = context.createTensor(workload.resource(1));

    const std::vector<float> input = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> expected = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F, 5.5F};
    writeMappedMemory(contextView.device, inputTensor.memory(), input);
    writeMappedMemory(contextView.device, outputTensor.memory(),
                      std::vector<float>(Tensor::numElements(tensorShape), 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{inputTensor.handle(), inputTensor.memory()});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{outputTensor.handle(), outputTensor.memory()});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(readMappedMemory<float>(contextView.device, outputTensor.memory(), Tensor::numElements(tensorShape)),
              expected);
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RecordAddConstantInputWithCallerOwnedContextAndTensors) {
    const std::vector<int64_t> tensorShape = {1, 3, 2, 1};

    const std::vector<float> constant(Tensor::numElements(tensorShape), 0.5F);
    auto workload = Workload::fromDataGraph(makeAddF32ConstantDescription(constant));
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR32Sfloat, tensorShape);
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR32Sfloat, tensorShape);

    const std::vector<float> input = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> expected = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F, 5.5F};
    writeMappedMemory(device, boundMemoryInfo(inputTensor), input);
    writeMappedMemory(device, boundMemoryInfo(outputTensor), std::vector<float>(Tensor::numElements(tensorShape), 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor, boundMemoryInfo(inputTensor)});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor, boundMemoryInfo(outputTensor)});

    auto execution = session.prepare(bindings);
    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    auto commandBuffer =
        std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    execution.record(*commandBuffer);
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);

    EXPECT_EQ(readMappedMemory<float>(device, boundMemoryInfo(outputTensor), Tensor::numElements(tensorShape)),
              expected);
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RecordDataGraphWorkload) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 29);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

    auto execution = session.prepare(bindings);
    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    auto commandBuffer =
        std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    execution.record(*commandBuffer);
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputShape));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphRepeatedDifferentInput) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();
    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});
    auto execution = session.prepare(bindings);

    for (const auto seed : {3U, 41U, 67U, 89U, 113U}) {
        const auto input = makeMaxpoolInput(inputShape, seed);
        inputTensor.write(input);
        outputTensor.fill(0, outputTensor.numElements());
        execution.run();
        EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputShape));
    }
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RecordDataGraphRepeatedDifferentInput) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();
    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});
    auto execution = session.prepare(bindings);

    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    for (const auto seed : {5U, 29U, 53U}) {
        const auto input = makeMaxpoolInput(inputShape, seed);
        inputTensor.write(input);
        outputTensor.fill(0, outputTensor.numElements());

        auto commandBuffer =
            std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
        const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

        commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        execution.record(*commandBuffer);
        commandBuffer.end();

        const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
        queue.submit(submitInfo, *fence);
        ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);
        EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputShape));
    }
}

TEST_F(StandaloneDataGraphSessionExecutionTest, PrepareMultipleExecutionsFromOneConfiguredSession) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor firstOutputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);
    const Tensor secondInputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor secondOutputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto firstInput = makeMaxpoolInput(inputShape, 7);
    const auto secondInput = makeMaxpoolInput(inputShape, 31);
    firstInputTensor.write(firstInput);
    firstOutputTensor.fill(0, firstOutputTensor.numElements());
    secondInputTensor.write(secondInput);
    secondOutputTensor.fill(0, secondOutputTensor.numElements());

    Session session(context, workload);
    session.configure();

    auto firstBindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    firstBindings.bindTensor(workload.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    firstBindings.bindTensor(workload.resource(1), TensorBindingInfo{*firstOutputTensor.tensor});
    auto firstExecution = session.prepare(firstBindings);

    auto secondBindings = session.createBindingSet();
    secondBindings.bindTensor(workload.resource(0), TensorBindingInfo{*secondInputTensor.tensor});
    secondBindings.bindTensor(workload.resource(1), TensorBindingInfo{*secondOutputTensor.tensor});
    auto secondExecution = session.prepare(secondBindings);

    firstExecution.run();
    secondExecution.run();

    EXPECT_EQ(firstOutputTensor.read(firstOutputTensor.numElements()), expectedMaxpool(firstInput, inputShape));
    EXPECT_EQ(secondOutputTensor.read(secondOutputTensor.numElements()), expectedMaxpool(secondInput, inputShape));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWithBorrowedConstant) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    const std::vector<int8_t> zeroWeights(16UL * 2 * 2 * 16, 0);
    const std::vector<int8_t> oneWeights(16UL * 2 * 2 * 16, 1);
    auto zeroWeightWorkload = Workload::fromDataGraph(makeConv2dRescaleConstantDescription(zeroWeights));
    auto oneWeightWorkload = Workload::fromDataGraph(makeConv2dRescaleConstantDescription(oneWeights));
    auto context = wrappedContext();

    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 7);
    inputTensor.write(input);

    auto run = [&](Workload &workload) {
        outputTensor.fill(0, outputTensor.numElements());

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        EXPECT_EQ(workload.resourceCount(), 2);
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

        auto execution = session.prepare(bindings);
        execution.run();
        return outputTensor.read(outputTensor.numElements());
    };

    const auto zeroWeightOutput = run(zeroWeightWorkload);
    const auto oneWeightOutput = run(oneWeightWorkload);

    EXPECT_NE(zeroWeightOutput, oneWeightOutput);
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWithSparseBorrowedConstant) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    constexpr std::size_t sparseWeightCount = std::size_t{16} * 2U * 2U * 16U;
    const auto sparseWeights = makeSparse2To4Weights(sparseWeightCount);
    auto denseWorkload = Workload::fromDataGraph(makeConv2dRescaleConstantDescription(sparseWeights));

    auto sparseDescription = makeConv2dRescaleConstantDescription(sparseWeights);
    ASSERT_EQ(sparseDescription.constants.size(), 1);
    sparseDescription.constants[0].sparse2To4 = DataGraphConstant::Sparsity{3};
    auto sparseWorkload = Workload::fromDataGraph(std::move(sparseDescription));

    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});
    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 23);
    inputTensor.write(input);

    auto run = [&](Workload &workload) {
        outputTensor.fill(0, outputTensor.numElements());

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        EXPECT_EQ(workload.resourceCount(), 2);
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

        auto execution = session.prepare(bindings);
        execution.run();
        return outputTensor.read(outputTensor.numElements());
    };

    EXPECT_EQ(run(sparseWorkload), run(denseWorkload));
}

/*******************************************************************************
 * Validation coverage
 *******************************************************************************/

TEST_F(StandaloneDataGraphSessionExecutionTest, RejectsRuntimeAllocationForWrongResourceKind) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();

    EXPECT_THROW((void)context.createBuffer(workload.resource(0)), std::runtime_error);
    EXPECT_THROW((void)context.createImage(workload.resource(0)), std::runtime_error);
}
