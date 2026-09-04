/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_standalone_compute_utils.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

} // namespace

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST(StandaloneComputeWorkload, ExposesPublicExecutablesBindingsResourcesAndDispatch) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());

    ASSERT_EQ(workload.executableCount(), 1);
    ASSERT_EQ(workload.resourceCount(), 3);
    ASSERT_EQ(workload.placeholderModuleCount(), 0);

    const auto executable = workload.executable(0);
    EXPECT_EQ(executable.index(), 0);
    EXPECT_EQ(executable.name(), "compute");
    EXPECT_EQ(executable.type(), ExecutableKind::Compute);
    const auto module = executable.module();
    EXPECT_EQ(module.index(), 0);
    EXPECT_EQ(module.name(), "compute");
    EXPECT_EQ(module.entryPoint(), "main");
    EXPECT_EQ(module.codeKind(), ModuleCodeKind::Spirv);

    ASSERT_EQ(executable.interfaceDescriptorBindingCount(), 3);
    const auto lhs = executable.interfaceDescriptorBinding(0);
    EXPECT_EQ(lhs.set, 0);
    EXPECT_EQ(lhs.binding, 0);
    EXPECT_EQ(lhs.resourceIndex, 0);
    EXPECT_EQ(lhs.access, ResourceAccess::Read);
    EXPECT_EQ(lhs.kind, ResourceKind::StorageBuffer);
    EXPECT_EQ(lhs.descriptorType, vk::DescriptorType::eStorageBuffer);

    const auto rhs = executable.interfaceDescriptorBinding(1);
    EXPECT_EQ(rhs.set, 0);
    EXPECT_EQ(rhs.binding, 1);
    EXPECT_EQ(rhs.resourceIndex, 1);
    EXPECT_EQ(rhs.access, ResourceAccess::Read);

    const auto output = executable.interfaceDescriptorBinding(2);
    EXPECT_EQ(output.set, 1);
    EXPECT_EQ(output.binding, 2);
    EXPECT_EQ(output.resourceIndex, 2);
    EXPECT_EQ(output.access, ResourceAccess::Write);

    const auto input = workload.resource(0);
    EXPECT_EQ(input.index(), 0);
    EXPECT_EQ(input.name(), "lhs");
    EXPECT_EQ(input.access(), ResourceAccess::Read);
    EXPECT_EQ(input.requirements().kind(), ResourceKind::StorageBuffer);
    EXPECT_EQ(input.requirements().descriptorType(), vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(input.requirements().format(), vk::Format::eR32Sint);
    EXPECT_EQ(input.requirements().elementCount(), 10);
    EXPECT_EQ(input.requirements().byteSize(), 10 * sizeof(int32_t));
    EXPECT_EQ(input.requirements().asBuffer().usage(), vk::BufferUsageFlagBits::eStorageBuffer);

    const auto result = workload.resource(2);
    EXPECT_EQ(result.name(), "output");
    EXPECT_EQ(result.access(), ResourceAccess::Write);
    EXPECT_EQ(result.requirements().kind(), ResourceKind::StorageBuffer);
    EXPECT_EQ(result.requirements().byteSize(), 10 * sizeof(int32_t));
}

TEST(StandaloneComputeWorkload, ReportsSourceModuleSupport) {
#ifdef ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT
    EXPECT_TRUE(supports(Feature::GlslModules));
#else
    EXPECT_FALSE(supports(Feature::GlslModules));
#endif

#ifdef ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT
    EXPECT_TRUE(supports(Feature::HlslModules));
#else
    EXPECT_FALSE(supports(Feature::HlslModules));
#endif
}

TEST(StandaloneComputeWorkload, AcceptsSpecializationConstants) {
    auto description = makeAddInt32BuffersDescription();
    description.specializationInfo.mapEntries = {vk::SpecializationMapEntry(0, 0, 4)};
    description.specializationInfo.data = {1, 2, 3, 4};

    EXPECT_NO_THROW((void)Workload::fromComputeShader(std::move(description)));
}

TEST(StandaloneComputeWorkload, AcceptsPushConstants) {
    auto description = makeAddInt32BuffersDescription();
    description.pushConstantSize = 4;

    EXPECT_NO_THROW((void)Workload::fromComputeShader(std::move(description)));
}

TEST(StandaloneComputeWorkload, ExposesImageSamplerRequirements) {
    auto description = makeAddInt32BuffersDescription();
    description.resources.push_back(
        {"runtime_sampler_image", 0, 3, ResourceAccess::Read, makeSampledImageRequirements(true)});
    description.resources.push_back(
        {"caller_sampler_image", 0, 4, ResourceAccess::Read, makeSampledImageRequirements(false)});

    auto workload = Workload::fromComputeShader(std::move(description));

    ASSERT_EQ(workload.resourceCount(), 5);

    const auto runtimeSamplerImage = workload.resource(3);
    const auto runtimeSamplerRequirements = runtimeSamplerImage.requirements().asImage();
    EXPECT_EQ(runtimeSamplerImage.name(), "runtime_sampler_image");
    EXPECT_EQ(runtimeSamplerRequirements.extent(), vk::Extent3D(2, 2, 1));
    EXPECT_TRUE(runtimeSamplerRequirements.isSampled());
    EXPECT_FALSE(runtimeSamplerRequirements.isStorage());
    EXPECT_TRUE(runtimeSamplerRequirements.hasRuntimeSampler());
    EXPECT_FALSE(runtimeSamplerRequirements.requiresSamplerBinding());
    EXPECT_EQ(runtimeSamplerRequirements.usage(), vk::ImageUsageFlagBits::eSampled);
    EXPECT_EQ(runtimeSamplerRequirements.requiredLayout(), vk::ImageLayout::eShaderReadOnlyOptimal);
    EXPECT_EQ(runtimeSamplerRequirements.requiredSubresourceRange().aspectMask, vk::ImageAspectFlagBits::eColor);

    const auto callerSamplerImage = workload.resource(4);
    const auto callerSamplerRequirements = callerSamplerImage.requirements().asImage();
    EXPECT_EQ(callerSamplerImage.name(), "caller_sampler_image");
    EXPECT_TRUE(callerSamplerRequirements.isSampled());
    EXPECT_FALSE(callerSamplerRequirements.hasRuntimeSampler());
    EXPECT_TRUE(callerSamplerRequirements.requiresSamplerBinding());
}

/*******************************************************************************
 * Validation coverage
 *******************************************************************************/

TEST(StandaloneComputeWorkload, RejectsInvalidPublicInspectionIndexes) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());

    EXPECT_THROW((void)workload.resource(workload.resourceCount()), std::out_of_range);
    EXPECT_THROW((void)workload.executable(workload.executableCount()), std::out_of_range);
    EXPECT_THROW((void)workload.placeholderModule(0), std::out_of_range);

    const auto executable = workload.executable(0);
    EXPECT_THROW((void)executable.interfaceDescriptorBinding(executable.interfaceDescriptorBindingCount()),
                 std::out_of_range);
}

TEST(StandaloneComputeWorkload, RejectsMismatchedResourceRequirementViews) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());

    const auto requirements = workload.resource(0).requirements();

    EXPECT_NO_THROW((void)requirements.asBuffer());
    EXPECT_THROW((void)requirements.asTensor(), std::runtime_error);
    EXPECT_THROW((void)requirements.asImage(), std::runtime_error);
}

TEST(StandaloneComputeWorkload, RejectsEmptySpecializationConstantPayload) {
    auto description = makeAddInt32BuffersDescription();
    description.specializationInfo.mapEntries = {vk::SpecializationMapEntry(0, 0, 4)};

    EXPECT_THROW((void)Workload::fromComputeShader(std::move(description)), std::runtime_error);
}

TEST(StandaloneComputeWorkload, RejectsMissingModuleCode) {
    auto description = makeAddInt32BuffersDescription();
    description.module.spirv.clear();

    EXPECT_THROW((void)Workload::fromComputeShader(std::move(description)), std::runtime_error);
}
