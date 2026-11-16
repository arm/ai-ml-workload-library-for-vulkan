/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/utils.hpp"
#include "test_expected_results.hpp"
#include "test_pipeline_creation.hpp"
#include "test_standalone_data_graph_utils.hpp"
#include "test_vulkan_fixture.hpp"
#include "test_vulkan_resources.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib_utils/application_context.hpp"
#include "mlworkloadlib_utils/mapped_device_memory.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_format_traits.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::test;

class StandaloneDataGraphSessionExecutionTest : public RuntimeSessionExecutionTest {};

#ifdef VK_ARM_tensors2
class ScaledGraphConstantExecutionTest : public StandaloneDataGraphSessionExecutionTest,
                                         public ::testing::WithParamInterface<vk::Format> {
    void SetUp() override {
        vk::PhysicalDeviceTensors2FeaturesARM tensorFeatures;
        tensorFeatures.tensorBlockScaled = true;
        RuntimeContextDeviceRequirements requirements;
        requirements.requiredDeviceExtensions = {VK_ARM_TENSORS_2_EXTENSION_NAME};
        requirements.deviceFeaturePNext = &tensorFeatures;
        if (!initialize(requirements)) {
            GTEST_SKIP() << "No Vulkan device with block-scaled data graph support";
        }
    }
};
#endif

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
 * Pipeline creation
 *******************************************************************************/

TEST(DataGraphPipelineCreateInfoTest, NeuralStatisticsPreserveShaderModuleCreateInfo) {
    vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleInfo;
    const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo({}, {}, {}, {}, &shaderModuleInfo);
    vk::DataGraphPipelineNeuralStatisticsCreateInfoARM neuralStatisticsInfo(vk::True);

    detail::insertPNextAfter(shaderModuleInfo, neuralStatisticsInfo);

    EXPECT_EQ(pipelineCreateInfo.pNext, &shaderModuleInfo);
    EXPECT_EQ(shaderModuleInfo.pNext, &neuralStatisticsInfo);
    EXPECT_EQ(neuralStatisticsInfo.pNext, nullptr);
}

#ifdef VK_ARM_tensors2
class ScaledGraphConstantSubmissionTest : public StandaloneDataGraphSessionExecutionTest,
                                          public ::testing::WithParamInterface<vk::Format> {};

TEST_P(ScaledGraphConstantSubmissionTest, SubmitsScaledConstantPlanes) {
    const auto format = GetParam();
    const auto valueSize = static_cast<size_t>(64U * vk::blockSize(format) / vk::blockExtent(format)[0]);
    std::vector<uint8_t> payload(valueSize + 2, 0x38);
    payload[valueSize] = 0x7f;     // E8M0 scale 1.
    payload[valueSize + 1] = 0x80; // E8M0 scale 2.
    const float bias = 0;
    auto description = makeMaxpoolDescription();
    auto requirements = tensorRequirements(format, {2, 32});
    requirements.tensor.scaledTensorConfig = ScaledTensorConfig{{32}, TensorScaleFormat::E8M0};
    description.constants = {
        {"bias", tensorRequirements(vk::Format::eR32Sfloat, {1}), &bias, sizeof(bias)},
        {"weights", requirements, payload.data(), payload.size()},
    };
    auto context = wrappedContext();
    auto workload = Workload::fromDataGraph(description);
    EXPECT_EQ(workload.resourceCount(), 2u);
    EXPECT_EQ(workload.executable(0).interfaceDescriptorBindingCount(), 2u);

    // Inspect submission independently of support for scaled shaders in the driver.
    PipelineCreationHook hook(device);
    hook.skipDriverCall = true;
    hook.inspectGraph = [&](const VkDataGraphPipelineCreateInfoARM &info) {
        const auto *node = static_cast<const VkBaseInStructure *>(info.pNext);
        while (node && node->sType != VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM) {
            node = node->pNext;
        }
        ASSERT_NE(node, nullptr);
        const auto *shader = reinterpret_cast<const VkDataGraphPipelineShaderModuleCreateInfoARM *>(node);
        ASSERT_EQ(shader->constantCount, 2u);
        EXPECT_EQ(info.resourceInfoCount, 2u);
        EXPECT_EQ(shader->pConstants[0].id, 0u);
        EXPECT_EQ(shader->pConstants[0].pConstantData, &bias);
        const auto *ordinary = static_cast<const VkTensorDescriptionARM *>(shader->pConstants[0].pNext);
        ASSERT_NE(ordinary, nullptr);
        EXPECT_EQ(ordinary->sType, VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM);
        EXPECT_EQ(ordinary->format, VK_FORMAT_R32_SFLOAT);

        const auto &constant = shader->pConstants[1];
        EXPECT_EQ(constant.id, 1u);
        EXPECT_EQ(constant.pConstantData, payload.data());
        const auto *scaled = static_cast<const VkBlockScaledTensorDescriptionARM *>(constant.pNext);
        ASSERT_NE(scaled, nullptr);
        ASSERT_EQ(scaled->sType, VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM);
        EXPECT_EQ(scaled->pNext, nullptr);
        EXPECT_EQ(scaled->blockScalingScheme, VK_TENSOR_BLOCK_SCALING_SCHEME_PLANAR_OCP_MICROSCALING_ARM);
        ASSERT_EQ(scaled->planeCount, 2u);
        EXPECT_EQ(scaled->pPlaneTypes[0], VK_TENSOR_PLANE_VALUES_ARM);
        EXPECT_EQ(scaled->pPlaneTypes[1], VK_TENSOR_PLANE_SCALES_ARM);
        for (uint32_t plane = 0; plane < 2; ++plane) {
            const auto &tensor = scaled->pPlaneDescriptions[plane];
            EXPECT_EQ(tensor.tiling, VK_TENSOR_TILING_LINEAR_ARM);
            EXPECT_EQ(tensor.usage, VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM);
            EXPECT_EQ(tensor.pStrides, nullptr);
            ASSERT_EQ(tensor.dimensionCount, 2u);
            EXPECT_EQ(tensor.pDimensions[0], 2);
            EXPECT_EQ(tensor.pDimensions[1], plane == 0 ? 32 : 1);
        }
        EXPECT_EQ(scaled->pPlaneDescriptions[0].format, static_cast<VkFormat>(format));
        EXPECT_EQ(scaled->pPlaneDescriptions[1].format, VK_FORMAT_R8_UFLOAT_FPENCODING_FLOAT8UNSIGNEDE8M0_ARM);
    };
    Session session(context, workload);
    EXPECT_THROW(session.configure(), vk::SystemError);
    ASSERT_EQ(hook.calls.size(), 1u);

    description.constants.back().size -= 1;
    auto truncated = Workload::fromDataGraph(std::move(description));
    Session truncatedSession(context, truncated);
    EXPECT_THROW(truncatedSession.configure(), std::runtime_error);
    EXPECT_EQ(hook.calls.size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(MXFormats, ScaledGraphConstantSubmissionTest,
                         ::testing::Values(vk::Format::eR8SfloatFpencodingFloat8E4M3ARM,
                                           vk::Format::eR8SfloatFpencodingFloat8E5M2ARM,
                                           vk::Format::eR8SfloatFpencodingMxint8ARM,
                                           vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat4E2M1ARM,
                                           vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat6E2M3ARM,
                                           vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat6E3M2ARM),
                         [](const ::testing::TestParamInfo<vk::Format> &info) {
                             switch (info.param) {
                             case vk::Format::eR8SfloatFpencodingFloat8E4M3ARM:
                                 return "E4M3";
                             case vk::Format::eR8SfloatFpencodingFloat8E5M2ARM:
                                 return "E5M2";
                             case vk::Format::eR8SfloatFpencodingMxint8ARM:
                                 return "MXINT8";
                             case vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat4E2M1ARM:
                                 return "FP4";
                             case vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat6E2M3ARM:
                                 return "FP6E2M3";
                             case vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat6E3M2ARM:
                                 return "FP6E3M2";
                             default:
                                 return "Unknown";
                             }
                         });
#endif

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST_F(StandaloneDataGraphSessionExecutionTest, RejectsPushConstantSizeOverrides) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();
    Session session(context, workload);
    EXPECT_EQ(session.pushConstantSize(workload.executable(0)), 0u);
    EXPECT_THROW(session.setPushConstantSize(workload.executable(0), 4), std::runtime_error);
    EXPECT_THROW(session.setPushConstantSize(workload.executable(0), 0), std::runtime_error);
}

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

TEST_F(StandaloneDataGraphSessionExecutionTest, ExecutableModuleReplacementUsesWorkloadEntryPoint) {
    auto description = makeMaxpoolDescription();
    auto implementation = description.module;
    // Preserve the interface and a non-default entry point. Missing original code
    // makes success depend on the executable-specific binding reaching the graph path.
    auto assembly = readSpvasmFile(ML_WORKLOAD_LIB_MAXPOOL_16X16_TO_8X8_SPVASM);
    replace(assembly, "INPUT_SET", "0");
    replace(assembly, "INPUT_BINDING", "0");
    replace(assembly, "OUTPUT_SET", "1");
    replace(assembly, "OUTPUT_BINDING", "1");
    replace(assembly, "\"main\"", "\"replacement_entry\"");
    implementation.spirv = assembleSpirv(assembly);
    description.entryPoint = "replacement_entry";
    description.module = {};
    auto workload = Workload::fromDataGraph(description);
    auto context = wrappedContext();
    Session session(context, workload);
    session.bindModule(workload.executable(0), std::move(implementation));
    session.configure();
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const Tensor input(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    const Tensor output(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const auto values = makeMaxpoolInput(inputShape, 11);
    input.write(values);
    output.fill(0, output.numElements());
    auto bindings = session.createBindingSet();
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*input.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*output.tensor});
    session.prepare(bindings).run();
    EXPECT_EQ(output.read(output.numElements()), expectedMaxpool(values, inputShape));
    EXPECT_TRUE(workload.executable(0).module().requiresImplementation());
    EXPECT_EQ(workload.executable(0).module().entryPoint(), "replacement_entry");
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

TEST_F(StandaloneDataGraphSessionExecutionTest, RejectNeuralStatisticsForProtectedAccessOnlyPipeline) {
    auto description = makeMaxpoolDescription();
    description.pipeline.flags = vk::PipelineCreateFlagBits2::eProtectedAccessOnly;
    auto workload = Workload::fromDataGraph(std::move(description));
    auto context = wrappedContext();

    SessionOptions options;
    options.neuralStatistics.emplace();
    Session session(context, workload, options);

    EXPECT_THROW(session.configure(), std::runtime_error);
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWorkloadWithPipelineSpecializationConstants) {
    auto context = wrappedContext();

    auto runArshift = [&](bool round, bool isOverride = false) {
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
        utils::writeDeviceMemory(device, boundMemoryInfo(inputTensor), input);
        utils::writeDeviceMemory(device, boundMemoryInfo(shiftTensor), shift);
        utils::writeDeviceMemory(device, boundMemoryInfo(outputTensor), std::vector<uint16_t>(input.size(), 0));

        Session session(context, workload);
        if (isOverride) {
            session.setSpecializationInfo(workload.executable(0),
                                          makeArshiftSpecBoolDescription(!round).pipeline.specializationInfo);
        }
        session.configure();
        EXPECT_EQ(session.specializationInfo(workload.executable(0)).data,
                  makeBoolSpecializationConstantData(isOverride ? !round : round));
        EXPECT_EQ(workload.executable(0).specializationInfo().data, makeBoolSpecializationConstantData(round));
        auto bindings = session.createBindingSet();
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        bindings.bindTensor(workload.resource(1), TensorBindingInfo{*shiftTensor.tensor});
        bindings.bindTensor(workload.resource(2), TensorBindingInfo{*outputTensor.tensor});

        auto execution = session.prepare(bindings);
        execution.run();

        return utils::readDeviceMemory<uint16_t>(device, boundMemoryInfo(outputTensor), input.size());
    };

    EXPECT_EQ(runArshift(false), (std::vector<uint16_t>{3, 4, 2, 3}));
    EXPECT_EQ(runArshift(true), (std::vector<uint16_t>{4, 5, 3, 4}));
    EXPECT_EQ(runArshift(false, true), (std::vector<uint16_t>{4, 5, 3, 4}));
    EXPECT_EQ(runArshift(true, true), (std::vector<uint16_t>{3, 4, 2, 3}));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, RunDataGraphWorkloadWithRuntimeCreatedContextAndTensors) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = Context::create();
    const auto contextView = context.contextView();
    Session session(context, workload);

    auto inputTensor = context.createTensor(workload.resource(0), session.tensorTiling(workload.resource(0)));
    auto outputTensor = context.createTensor(workload.resource(1), session.tensorTiling(workload.resource(1)));

    const auto input = makeMaxpoolInput(inputShape, 19);
    utils::writeDeviceMemory(contextView.device, inputTensor.memory(), input);
    utils::writeDeviceMemory(contextView.device, outputTensor.memory(),
                             std::vector<int8_t>(Tensor::numElements(outputShape), 0));

    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{inputTensor.handle(), inputTensor.memory()});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{outputTensor.handle(), outputTensor.memory()});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(
        utils::readDeviceMemory<int8_t>(contextView.device, outputTensor.memory(), Tensor::numElements(outputShape)),
        expectedMaxpool(input, inputShape));
}

TEST_F(StandaloneDataGraphSessionExecutionTest, MovesRuntimeOwnedTensorAllocations) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = wrappedContext();
    const Session session(context, workload);

    auto tensor = context.createTensor(workload.resource(0), session.tensorTiling(workload.resource(0)));
    auto movedTensor(std::move(tensor));
    auto assignedTensor = context.createTensor(workload.resource(1), session.tensorTiling(workload.resource(1)));
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
    Session session(context, workload);

    auto inputTensor = context.createTensor(workload.resource(0), session.tensorTiling(workload.resource(0)));
    auto outputTensor = context.createTensor(workload.resource(1), session.tensorTiling(workload.resource(1)));

    const std::vector<float> input = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> expected = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F, 5.5F};
    utils::writeDeviceMemory(contextView.device, inputTensor.memory(), input);
    utils::writeDeviceMemory(contextView.device, outputTensor.memory(),
                             std::vector<float>(Tensor::numElements(tensorShape), 0));

    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{inputTensor.handle(), inputTensor.memory()});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{outputTensor.handle(), outputTensor.memory()});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(
        utils::readDeviceMemory<float>(contextView.device, outputTensor.memory(), Tensor::numElements(tensorShape)),
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
    utils::writeDeviceMemory(device, boundMemoryInfo(inputTensor), input);
    utils::writeDeviceMemory(device, boundMemoryInfo(outputTensor),
                             std::vector<float>(Tensor::numElements(tensorShape), 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor, boundMemoryInfo(inputTensor)});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor, boundMemoryInfo(outputTensor)});

    auto execution = session.prepare(bindings);
    recordAndSubmitCommands(device, queue, queueFamilyIndex,
                            [&execution](vk::CommandBuffer commandBuffer) { execution.record(commandBuffer); });

    EXPECT_EQ(utils::readDeviceMemory<float>(device, boundMemoryInfo(outputTensor), Tensor::numElements(tensorShape)),
              expected);
}

#ifdef VK_ARM_tensors2
TEST_P(ScaledGraphConstantExecutionTest, RunsConv2dWithScaledWeights) {
    const auto format = GetParam();
    const bool isE4M3 = format == vk::Format::eR8SfloatFpencodingFloat8E4M3ARM;
    auto assembly = readSpvasmFile(ML_WORKLOAD_LIB_SCALED_CONSTANT_CONV2D_SPVASM);
    replace(assembly, "VALUE_ENCODING", isE4M3 ? "Float8E4M3EXT" : "Float8E5M2EXT");

    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleSpirv(assembly);
    description.entryPoint = "main";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, tensorRequirements(vk::Format::eR32Sfloat, {1, 1, 1, 64})},
        {"output", 0, 1, ResourceAccess::Write, tensorRequirements(vk::Format::eR32Sfloat, {1, 1, 1, 1})},
    };
    auto weights = tensorRequirements(format, {1, 1, 1, 64});
    weights.tensor.scaledTensorConfig = ScaledTensorConfig{{32}, TensorScaleFormat::E8M0};
    std::array<uint8_t, 66> payload{};
    std::fill_n(payload.begin(), 64, isE4M3 ? 0x38 : 0x3c);
    payload[64] = 0x7f; // E8M0 scale 1.
    payload[65] = 0x80; // E8M0 scale 2.
    const float bias = 3.0F;
    description.constants = {
        {"weights", weights, payload.data(), payload.size()},
        {"bias", tensorRequirements(vk::Format::eR32Sfloat, {1}), &bias, sizeof(bias)},
    };
    auto workload = Workload::fromDataGraph(description);
    auto context = wrappedContext();
    ASSERT_EQ(workload.resourceCount(), 2u);
    ASSERT_EQ(workload.executable(0).interfaceDescriptorBindingCount(), 2u);

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR32Sfloat, {1, 1, 1, 64});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR32Sfloat, {1, 1, 1, 1});
    std::vector<float> input(64, 3.0F);
    std::fill_n(input.begin(), 32, 1.0F);
    utils::writeDeviceMemory(device, boundMemoryInfo(inputTensor), input);
    utils::writeDeviceMemory(device, boundMemoryInfo(outputTensor), std::vector<float>{0.0F});

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor, boundMemoryInfo(inputTensor)});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor, boundMemoryInfo(outputTensor)});
    auto execution = session.prepare(bindings);
    execution.run();

    // 32 * 1 * 1 + 32 * 3 * 2 + 3 = 227, including the ordinary bias.
    EXPECT_EQ(utils::readDeviceMemory<float>(device, boundMemoryInfo(outputTensor), 1), std::vector<float>{227.0F});
}

INSTANTIATE_TEST_SUITE_P(FP8Formats, ScaledGraphConstantExecutionTest,
                         ::testing::Values(vk::Format::eR8SfloatFpencodingFloat8E4M3ARM,
                                           vk::Format::eR8SfloatFpencodingFloat8E5M2ARM),
                         [](const ::testing::TestParamInfo<vk::Format> &info) {
                             return info.param == vk::Format::eR8SfloatFpencodingFloat8E4M3ARM ? "E4M3" : "E5M2";
                         });
#endif

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
    recordAndSubmitCommands(device, queue, queueFamilyIndex,
                            [&execution](vk::CommandBuffer commandBuffer) { execution.record(commandBuffer); });

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

        recordAndSubmitCommands(device, queue, *commandPool,
                                [&execution](vk::CommandBuffer commandBuffer) { execution.record(commandBuffer); });
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

TEST_F(StandaloneDataGraphSessionExecutionTest, RunTwoSessionsWithSharedCompiledExecution) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});
    CompiledExecution compiledExecution(context, workload);

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor firstOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor secondInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor secondOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});

    const auto firstInput = makeMaxpoolInput(firstInputTensor.shape, 9);
    const auto secondInput = makeMaxpoolInput(secondInputTensor.shape, 17);
    firstInputTensor.write(firstInput);
    firstOutputTensor.fill(0, firstOutputTensor.numElements());
    secondInputTensor.write(secondInput);
    secondOutputTensor.fill(0, secondOutputTensor.numElements());

    Session firstSession(context, workload, compiledExecution);
    firstSession.configure();
    auto firstBindings = firstSession.createBindingSet();
    firstBindings.bindTensor(workload.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    firstBindings.bindTensor(workload.resource(1), TensorBindingInfo{*firstOutputTensor.tensor});
    auto firstExecution = firstSession.prepare(firstBindings);

    Session secondSession(context, workload, compiledExecution);
    secondSession.configure();
    auto secondBindings = secondSession.createBindingSet();
    secondBindings.bindTensor(workload.resource(0), TensorBindingInfo{*secondInputTensor.tensor});
    secondBindings.bindTensor(workload.resource(1), TensorBindingInfo{*secondOutputTensor.tensor});
    auto secondExecution = secondSession.prepare(secondBindings);

    firstExecution.run();
    secondExecution.run();

    EXPECT_EQ(firstOutputTensor.read(firstOutputTensor.numElements()),
              expectedMaxpool(firstInput, firstInputTensor.shape));
    EXPECT_EQ(secondOutputTensor.read(secondOutputTensor.numElements()),
              expectedMaxpool(secondInput, secondInputTensor.shape));
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
    EXPECT_THROW((void)context.createImage(workload.resource(0), vk::ImageTiling::eOptimal), std::runtime_error);
}

/*******************************************************************************
 * Data graph binary inputs
 *******************************************************************************/

#ifdef VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME
struct NullVulkanContext : utils::ApplicationContext {
    Context wrap() const { return Context::wrap({instance, physicalDevice, device, 0, queue}); }
};

constexpr std::array<std::byte, 17> binaryWithValidHeader = {
    std::byte{0xff}, std::byte{0xdd}, std::byte{0x22}, std::byte{0x11}, std::byte{0},   std::byte{0},
    std::byte{0},    std::byte{0},    std::byte{1},    std::byte{0},    std::byte{0},   std::byte{0},
    std::byte{0},    std::byte{0},    std::byte{0},    std::byte{0},    std::byte{0xaa}};
constexpr ArrayView<const std::byte> validBinary(binaryWithValidHeader.data(), binaryWithValidHeader.size());

TEST_F(StandaloneDataGraphSessionExecutionTest, BinaryPipelinePolicyAndFeedback) {
    auto context = wrappedContext();
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    // Supply alignment for the dummy binary and restore the property query on every exit.
    ScopedDispatcherHook<PFN_vkGetPhysicalDeviceProperties2> properties(
        physicalDevice, &vk::raii::detail::InstanceDispatcher::vkGetPhysicalDeviceProperties2,
        +[](VkPhysicalDevice, VkPhysicalDeviceProperties2 *info) {
            auto *binary = static_cast<VkPhysicalDeviceDataGraphBinariesPropertiesARM *>(info->pNext);
            binary->dataGraphPipelineBinaryDataAlignment = 1;
        });
    for (bool policy : {false, true}) {
        PipelineCreationHook hook(device);
        hook.skipDriverCall = true;
        hook.result = VK_PIPELINE_COMPILE_REQUIRED;
        SessionOptions options;
        options.failOnCompileRequired = policy;
        options.collectPipelineCreationFeedback = true;
        Session session(context, workload, options);
        session.bindDataGraphBinary(validBinary);
        EXPECT_THROW(session.configure(), vk::SystemError);
        ASSERT_EQ(hook.calls.size(), 1u);
        const auto &call = hook.calls[0];
        EXPECT_EQ(call.failOnCompileRequired, policy);
        EXPECT_TRUE(call.hasBinary);
        EXPECT_TRUE(call.hasFeedback);
        EXPECT_EQ(call.stageFeedbackCount, 0u);
    }
}

TEST(DataGraphBinary, RejectsComputeWorkload) {
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    ComputeShaderDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = {0x07230203};
    auto workload = Workload::fromComputeShader(std::move(description));
    Session session(context, workload);
    const std::byte byte{1};

    EXPECT_THROW(session.bindDataGraphBinary({&byte, 1}), std::runtime_error);
}

TEST(DataGraphBinary, RejectsMalformedHeader) {
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());
    Session session(context, workload);

    const auto expectError = [&](const std::vector<std::byte> &bytes, const char *message) {
        try {
            session.bindDataGraphBinary({bytes.data(), bytes.size()});
            FAIL() << "Data graph binary header was accepted";
        } catch (const std::runtime_error &error) {
            EXPECT_STREQ(message, error.what());
        }
    };

    std::vector<std::byte> bytes(binaryWithValidHeader.begin(), binaryWithValidHeader.end());
    bytes.resize(15);
    expectError(bytes, "Data graph binary header is truncated");

    bytes.assign(binaryWithValidHeader.begin(), binaryWithValidHeader.end());
    bytes[0] = std::byte{0};
    expectError(bytes, "Data graph binary header has an invalid magic value");

    bytes.assign(binaryWithValidHeader.begin(), binaryWithValidHeader.end());
    bytes[8] = std::byte{2};
    expectError(bytes, "Data graph binary payload size does not match its header");
}

TEST(DataGraphBinary, BindsStandaloneGraph) {
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    auto workload = Workload::fromDataGraph(DataGraphDescription{});
    Session session(context, workload);
    auto bytes = binaryWithValidHeader;
    bytes[4] = std::byte{1};
    const std::array<uint8_t, 2> identifier = {0, 0xff};

    EXPECT_EQ(workload.executable(0).module().codeKind(), ModuleCodeKind::Missing);
    EXPECT_NO_THROW(session.bindDataGraphBinary({bytes.data(), bytes.size()},
                                                DataGraphBinaryStorageMode::ApplicationBacked,
                                                {identifier.data(), identifier.size()}));
}

TEST(DataGraphBinary, RejectsExecutableModuleConflictsInBothCallOrders) {
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    auto workload = Workload::fromDataGraph(DataGraphDescription{});
    ModuleImplementation implementation;
    implementation.codeKind = ModuleCodeKind::Spirv;
    implementation.spirv = {0x07230203};
    Session moduleFirst(context, workload);
    moduleFirst.bindModule(workload.executable(0), implementation);
    EXPECT_THROW(moduleFirst.bindDataGraphBinary(validBinary), std::runtime_error);
    Session binaryFirst(context, workload);
    binaryFirst.bindDataGraphBinary(validBinary);
    EXPECT_THROW(binaryFirst.bindModule(workload.executable(0), implementation), std::runtime_error);
}

TEST(DataGraphBinary, RejectsSpecializationConflictsInBothCallOrders) {
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    auto workload = Workload::fromDataGraph(DataGraphDescription{});
    const auto step = workload.executable(0);
    const auto specialization = makeArshiftSpecBoolDescription(true).pipeline.specializationInfo;
    Session specializationFirst(context, workload);
    specializationFirst.setSpecializationInfo(step, specialization);
    EXPECT_THROW(specializationFirst.bindDataGraphBinary(validBinary), std::runtime_error);
    specializationFirst.setSpecializationInfo(step, {});
    EXPECT_NO_THROW(specializationFirst.bindDataGraphBinary(validBinary));
    Session binaryFirst(context, workload);
    binaryFirst.bindDataGraphBinary(validBinary);
    EXPECT_THROW(binaryFirst.setSpecializationInfo(step, specialization), std::runtime_error);
    EXPECT_NO_THROW(binaryFirst.setSpecializationInfo(step, {}));
}

TEST(DataGraphBinary, RejectsConflictingCompilationInputs) {
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    auto workload = Workload::fromDataGraph(DataGraphDescription{});
    ASSERT_EQ(workload.placeholderModuleCount(), 1);

    const auto bindModule = [&workload](Session &session) {
        ModuleImplementation implementation;
        implementation.codeKind = ModuleCodeKind::Spirv;
        implementation.spirv = {0x07230203};
        session.bindModule(workload.placeholderModule(0), std::move(implementation));
    };

    Session moduleFirst(context, workload);
    bindModule(moduleFirst);
    EXPECT_THROW(moduleFirst.bindDataGraphBinary(validBinary), std::runtime_error);

    Session binaryFirst(context, workload);
    binaryFirst.bindDataGraphBinary(validBinary);
    EXPECT_THROW(bindModule(binaryFirst), std::runtime_error);
}
#endif
