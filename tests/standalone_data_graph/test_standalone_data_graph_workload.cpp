/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_standalone_data_graph_utils.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

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

    const auto output = workload.resource(1);
    EXPECT_EQ(output.name(), "output");
    EXPECT_EQ(output.access(), ResourceAccess::Write);
    EXPECT_EQ(output.requirements().kind(), ResourceKind::Tensor);
    EXPECT_EQ(output.requirements().elementCount(), 1 * 8 * 8 * 16);
}

TEST(StandaloneDataGraphWorkload, ExposesPipelineMetadata) {
    auto description = makeMaxpoolDescription();
    description.pipeline.identifier = "maxpool-pipeline";
    description.pipeline.flags = vk::PipelineCreateFlagBits2::eDisableOptimization;
    description.pipeline.specializationInfo.mapEntries = {vk::SpecializationMapEntry(7, 0, 4)};
    description.pipeline.specializationInfo.data = {1, 0, 0, 0};

    auto workload = Workload::fromDataGraph(std::move(description));

    const auto metadata = workload.executable(0).dataGraphPipelineMetadata();
    EXPECT_EQ(metadata.identifier(), "maxpool-pipeline");
    EXPECT_EQ(metadata.flags(), vk::PipelineCreateFlagBits2::eDisableOptimization);
    ASSERT_EQ(metadata.specializationMapEntries().size(), 1);
    EXPECT_EQ(metadata.specializationMapEntries()[0].constantID, 7);
    EXPECT_EQ(metadata.specializationMapEntries()[0].offset, 0);
    EXPECT_EQ(metadata.specializationMapEntries()[0].size, 4);
    EXPECT_EQ(std::vector<uint8_t>(metadata.specializationData().begin(), metadata.specializationData().end()),
              (std::vector<uint8_t>{1, 0, 0, 0}));
}

TEST(StandaloneDataGraphWorkload, AcceptsSparseConstantsWithDimension) {
    std::array<int32_t, 1> constantData = {1};
    auto description = makeMaxpoolDescription();
    description.constants.push_back({"constant", makeTensorRequirements(vk::Format::eR32Sint, {1}), constantData.data(),
                                     sizeof(int32_t), DataGraphConstant::Sparsity{0}});

    EXPECT_NO_THROW((void)Workload::fromDataGraph(std::move(description)));
}

TEST(StandaloneDataGraphWorkload, HidesConstantsFromPublicResources) {
    std::array<int32_t, 4> constantData = {1, 2, 3, 4};
    auto description = makeMaxpoolDescription();
    description.constants.push_back({"constant", makeTensorRequirements(vk::Format::eR32Sint, {4}), constantData.data(),
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
    description.pipeline.specializationInfo.data = {1, 0, 0, 0, 0, 0, 0, 0};

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, RejectsSparseConstantsWithoutDimension) {
    std::array<int32_t, 1> constantData = {1};
    auto description = makeMaxpoolDescription();
    description.constants.push_back({"constant", makeTensorRequirements(vk::Format::eR32Sint, {1}), constantData.data(),
                                     sizeof(int32_t), DataGraphConstant::Sparsity{-1}});

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}

TEST(StandaloneDataGraphWorkload, RejectsNullConstantPayload) {
    auto description = makeMaxpoolDescription();
    description.constants.push_back(
        {"constant", makeTensorRequirements(vk::Format::eR32Sint, {1}), nullptr, sizeof(int32_t)});

    EXPECT_THROW((void)Workload::fromDataGraph(std::move(description)), std::runtime_error);
}
