/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_standalone_data_graph_utils.hpp"

#include "internal/workload_impl.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::test;

} // namespace

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST(StandaloneDataGraphWorkload, ExposesTensorResourcesAndExecutableMetadata) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());

    ASSERT_EQ(workload.executableCount(), 1);
    ASSERT_EQ(workload.resourceCount(), 2);
    ASSERT_EQ(workload.placeholderModuleCount(), 0);

    const auto executable = workload.executable(0);
    EXPECT_EQ(executable.index(), 0);
    EXPECT_EQ(executable.name(), "data_graph");
    EXPECT_EQ(executable.type(), ExecutableKind::Graph);
    const auto module = executable.module();
    EXPECT_EQ(module.index(), 0);
    EXPECT_EQ(module.name(), "data_graph");
    EXPECT_EQ(module.entryPoint(), "main");
    EXPECT_EQ(module.codeKind(), ModuleCodeKind::Spirv);

    ASSERT_EQ(executable.interfaceDescriptorBindingCount(), 2);
    const auto inputBinding = executable.interfaceDescriptorBinding(0);
    EXPECT_EQ(inputBinding.set, 0);
    EXPECT_EQ(inputBinding.binding, 0);
    EXPECT_EQ(inputBinding.resourceIndex, 0);
    EXPECT_EQ(inputBinding.access, ResourceAccess::Read);
    EXPECT_EQ(inputBinding.kind, ResourceKind::Tensor);
    EXPECT_EQ(inputBinding.descriptorType, vk::DescriptorType::eTensorARM);

    const auto outputBinding = executable.interfaceDescriptorBinding(1);
    EXPECT_EQ(outputBinding.set, 1);
    EXPECT_EQ(outputBinding.binding, 1);
    EXPECT_EQ(outputBinding.resourceIndex, 1);
    EXPECT_EQ(outputBinding.access, ResourceAccess::Write);

    const auto input = workload.resource(0);
    const auto inputRequirements = input.requirements();
    EXPECT_EQ(input.index(), 0);
    EXPECT_EQ(input.name(), "input");
    EXPECT_EQ(input.access(), ResourceAccess::Read);
    EXPECT_EQ(inputRequirements.kind(), ResourceKind::Tensor);
    EXPECT_EQ(inputRequirements.descriptorType(), vk::DescriptorType::eTensorARM);
    EXPECT_EQ(inputRequirements.format(), vk::Format::eR8Sint);
    EXPECT_EQ(inputRequirements.elementCount(), 1 * 16 * 16 * 16);
    EXPECT_EQ(inputRequirements.byteSize(), 1 * 16 * 16 * 16);
    EXPECT_EQ(inputRequirements.asTensor().usage(), vk::TensorUsageFlagBitsARM::eDataGraph);
    const auto inputShape = inputRequirements.asTensor().shape();
    EXPECT_EQ(inputShape.size(), 4);
    EXPECT_EQ(std::vector<int64_t>(inputShape.begin(), inputShape.end()), (std::vector<int64_t>{1, 16, 16, 16}));
    EXPECT_TRUE(inputRequirements.asTensor().stride().empty());
    EXPECT_TRUE(inputRequirements.asTensor().blockShape().empty());
    EXPECT_FALSE(inputRequirements.asTensor().scaleFormat());
    EXPECT_TRUE(inputRequirements.asTensor().scaleShape().empty());

    const auto output = workload.resource(1);
    EXPECT_EQ(output.name(), "output");
    EXPECT_EQ(output.access(), ResourceAccess::Write);
    EXPECT_EQ(output.requirements().kind(), ResourceKind::Tensor);
    EXPECT_EQ(output.requirements().elementCount(), 1 * 8 * 8 * 16);
}

TEST(StandaloneDataGraphWorkload, ExposesPipelineMetadata) {
    const std::vector<std::byte> specializationData = {std::byte{0x01}, std::byte{0x80}, std::byte{0xFE},
                                                       std::byte{0xFF}};
    auto description = makeMaxpoolDescription();
    description.pipeline.identifier = "maxpool-pipeline";
    description.pipeline.flags = vk::PipelineCreateFlagBits2::eDisableOptimization;
    description.pipeline.specializationInfo.mapEntries = {vk::SpecializationMapEntry(7, 0, 4)};
    description.pipeline.specializationInfo.data = specializationData;

    auto workload = Workload::fromDataGraph(std::move(description));

    const auto metadata = workload.executable(0).dataGraphPipelineMetadata();
    EXPECT_EQ(metadata.identifier(), "maxpool-pipeline");
    EXPECT_EQ(metadata.flags(), vk::PipelineCreateFlagBits2::eDisableOptimization);
    ASSERT_EQ(metadata.specializationMapEntries().size(), 1);
    EXPECT_EQ(metadata.specializationMapEntries()[0].constantID, 7);
    EXPECT_EQ(metadata.specializationMapEntries()[0].offset, 0);
    EXPECT_EQ(metadata.specializationMapEntries()[0].size, 4);
    const ArrayView<const std::byte> data = metadata.specializationData();
    EXPECT_EQ(std::vector<std::byte>(data.begin(), data.end()), specializationData);
    EXPECT_EQ(data.data(), workload.executable(0).dataGraphPipelineMetadata().specializationData().data());
}

TEST(StandaloneDataGraphWorkload, AcceptsSparseConstantsWithDimension) {
    std::array<int32_t, 1> constantData = {1};
    auto description = makeMaxpoolDescription();
    description.constants.push_back({"constant", tensorRequirements(vk::Format::eR32Sint, {1}), constantData.data(),
                                     sizeof(int32_t), DataGraphConstant::Sparsity{0}});

    EXPECT_NO_THROW((void)Workload::fromDataGraph(std::move(description)));
}

TEST(StandaloneDataGraphWorkload, HidesConstantsFromPublicResources) {
    std::array<int32_t, 4> constantData = {1, 2, 3, 4};
    auto description = makeMaxpoolDescription();
    description.constants.push_back({"constant", tensorRequirements(vk::Format::eR32Sint, {4}), constantData.data(),
                                     constantData.size() * sizeof(int32_t)});

    auto workload = Workload::fromDataGraph(std::move(description));

    ASSERT_EQ(workload.executableCount(), 1);
    EXPECT_EQ(workload.resourceCount(), 2);
    EXPECT_EQ(workload.resource(0).name(), "input");
    EXPECT_EQ(workload.resource(1).name(), "output");
    EXPECT_EQ(workload.executable(0).interfaceDescriptorBindingCount(), 2);
}

/*******************************************************************************
 * Validation coverage
 *******************************************************************************/

TEST(StandaloneDataGraphWorkload, RejectsMismatchedResourceRequirementViews) {
    auto workload = Workload::fromDataGraph(makeMaxpoolDescription());

    const auto requirements = workload.resource(0).requirements();

    EXPECT_NO_THROW((void)requirements.asTensor());
    EXPECT_THROW((void)requirements.asBuffer(), std::runtime_error);
    EXPECT_THROW((void)requirements.asImage(), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, RejectsEmptyPipelineSpecializationConstantPayload) {
    auto description = makeMaxpoolDescription();
    description.pipeline.specializationInfo.mapEntries = {vk::SpecializationMapEntry(0, 0, 4)};

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, RejectsDuplicatePipelineSpecializationConstantIds) {
    auto description = makeMaxpoolDescription();
    description.pipeline.specializationInfo.mapEntries = {vk::SpecializationMapEntry(0, 0, 4),
                                                          vk::SpecializationMapEntry(0, 4, 4)};
    description.pipeline.specializationInfo.data = {std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0},
                                                    std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}};

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, RejectsSparseConstantsWithoutDimension) {
    std::array<int32_t, 1> constantData = {1};
    auto description = makeMaxpoolDescription();
    description.constants.push_back({"constant", tensorRequirements(vk::Format::eR32Sint, {1}), constantData.data(),
                                     sizeof(int32_t), DataGraphConstant::Sparsity{-1}});

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, RejectsNullConstantPayload) {
    auto description = makeMaxpoolDescription();
    description.constants.push_back(
        {"constant", tensorRequirements(vk::Format::eR32Sint, {1}), nullptr, sizeof(int32_t)});

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, ReportsScaledTensorAndConstantRequirements) {
    for (const uint32_t block : {16u, 32u}) {
        auto description = makeMaxpoolDescription();
        description.resources.resize(1);
        auto &requirements = description.resources[0].resource;
        requirements = tensorRequirements(vk::Format::eR8SfloatFpencodingFloat8E4M3ARM, {2, 64});
        requirements.tensor.scaledTensorConfig = ScaledTensorConfig{{block}, TensorScaleFormat::E8M0};
        const std::array<uint8_t, 128> values{};
        description.constants = {{"scaled_constant", requirements, values.data(), values.size()}};
        auto workload = Workload::fromDataGraph(std::move(description));
        ASSERT_EQ(workload.resourceCount(), 1u);
        EXPECT_EQ(workload.executable(0).interfaceDescriptorBindingCount(), 1u);
        const auto resource = workload.resource(0).requirements();
        EXPECT_EQ(resource.format(), vk::Format::eR8SfloatFpencodingFloat8E4M3ARM);
        const auto tensor = resource.asTensor();
        EXPECT_EQ(tensor.scaleFormat(), TensorScaleFormat::E8M0);
        ASSERT_EQ(tensor.blockShape().size(), 1u);
        EXPECT_EQ(tensor.blockShape()[0], block);
        ASSERT_EQ(tensor.scaleShape().size(), 2u);
        EXPECT_EQ(tensor.scaleShape()[0], 2);
        EXPECT_EQ(tensor.scaleShape()[1], 64 / block);
        const auto &state = workloadImpl(workload);
        ASSERT_EQ(state.constants.size(), 1u);
        const auto &constant = state.resources.at(state.constants[0].resourceIndex);
        EXPECT_EQ(constant.format, resource.format());
        const auto &metadata = detail::tensorMetadata(constant);
        ASSERT_TRUE(metadata.scaledTensorConfig);
        EXPECT_EQ(metadata.scaledTensorConfig->blockShape, (std::vector<uint32_t>{block}));
        EXPECT_EQ(metadata.scaledTensorConfig->scaleFormat, TensorScaleFormat::E8M0);
        EXPECT_EQ(metadata.scaleShape, (std::vector<int64_t>{2, 64 / block}));
    }
}

TEST(StandaloneDataGraphWorkload, RejectsInvalidScaledTensorMetadata) {
    const std::array<std::vector<uint32_t>, 4> invalidBlocks = {{{}, {0}, {3}, {16, 1}}};
    for (const auto &block : invalidBlocks) {
        auto description = makeMaxpoolDescription();
        description.resources[0].resource.format = vk::Format::eR8SfloatFpencodingFloat8E4M3ARM;
        auto &tensor = description.resources[0].resource.tensor;
        tensor.shape = {64};
        tensor.scaledTensorConfig = ScaledTensorConfig{block, TensorScaleFormat::E8M0};
        EXPECT_THROW(Workload::fromDataGraph(std::move(description)), std::runtime_error);
    }
    auto description = makeMaxpoolDescription();
    description.resources[0].resource.format = vk::Format::eR8SfloatFpencodingFloat8E4M3ARM;
    description.resources[0].resource.tensor.scaledTensorConfig =
        ScaledTensorConfig{{1, 1, 1, 16}, static_cast<TensorScaleFormat>(-1)};
    EXPECT_THROW(Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, PreservesUnresolvedScaledTensorDimensions) {
    for (const size_t unresolvedAxis : {0u, 1u}) {
        std::vector<int64_t> shape{2, 64};
        std::vector<int64_t> expectedScaleShape{2, 4};
        shape[unresolvedAxis] = -1;
        expectedScaleShape[unresolvedAxis] = -1;
        auto description = makeMaxpoolDescription();
        description.resources.resize(1);
        auto &requirements = description.resources[0].resource;
        requirements = tensorRequirements(vk::Format::eR8SfloatFpencodingFloat8E4M3ARM, shape);
        requirements.tensor.scaledTensorConfig = ScaledTensorConfig{{16}, TensorScaleFormat::E8M0};
        const auto workload = Workload::fromDataGraph(std::move(description));
        const auto reported = workload.resource(0).requirements();
        EXPECT_EQ(reported.elementCount(), 0u);
        EXPECT_EQ(reported.byteSize(), 0u);
        const auto scaleShape = reported.asTensor().scaleShape();
        EXPECT_EQ(std::vector<int64_t>(scaleShape.begin(), scaleShape.end()), expectedScaleShape);
    }
}

TEST(StandaloneDataGraphWorkload, RejectsUnsupportedScaledValueFormats) {
    for (const auto format : {vk::Format::eR8Sint, vk::Format::eR32Sfloat, vk::Format::eUndefined}) {
        for (const bool constant : {false, true}) {
            SCOPED_TRACE(static_cast<uint32_t>(format));
            SCOPED_TRACE(constant);
            auto description = makeMaxpoolDescription();
            auto requirements = tensorRequirements(format, {32});
            requirements.tensor.scaledTensorConfig = ScaledTensorConfig{{32}, TensorScaleFormat::E8M0};
            const std::array<uint8_t, 128> values{};
            if (constant) {
                description.constants = {{"constant", requirements, values.data(), values.size()}};
            } else {
                description.resources[0].resource = requirements;
            }
            EXPECT_THAT([&] { Workload::fromDataGraph(description); },
                        testing::ThrowsMessage<std::runtime_error>(
                            testing::HasSubstr("Unsupported scaled tensor value format")));
        }
    }
}
