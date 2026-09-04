/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlworkloadlib/workload.hpp"
#include "test_vgf_utils.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mlsdk::vgflib;
using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

const std::vector<GraphConstantBindingRef> noGraphConstants;

} // namespace

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST(VgfWorkload, DecodesPublicExecutablesBindingsResourcesAndDispatch) {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 0, 1});
    const auto data = writeVgf([&](Encoder &encoder) {
        const auto module = encoder.AddModule(ModuleType::COMPUTE, "shader", "main", code);
        const auto input = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SFLOAT, {4}, {4});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {8}, {});
        const auto intermediate =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {8}, {});
        const auto inputBinding = encoder.AddBindingSlot(3, input);
        const auto outputBinding = encoder.AddBindingSlot(5, output);
        const auto intermediateBinding = encoder.AddBindingSlot(7, intermediate);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, outputBinding, intermediateBinding}, 2);
        encoder.AddSegmentInfo(module, "executable", {descriptorSet}, {inputBinding}, {outputBinding}, noGraphConstants,
                               {1, 2, 3});
    });

    auto workload = Workload::fromVGF(data.data(), data.size());

    ASSERT_EQ(workload.executableCount(), 1);
    ASSERT_EQ(workload.resourceCount(), 2);
    ASSERT_EQ(workload.placeholderModuleCount(), 0);

    std::vector<ResourceAccess> resourceAccesses;
    for (const auto resource : workload.resources()) {
        resourceAccesses.push_back(resource.access());
    }
    EXPECT_EQ(resourceAccesses, (std::vector<ResourceAccess>{ResourceAccess::Read, ResourceAccess::Write}));

    uint32_t rangedExecutableCount = 0;
    for (const auto rangedExecutable : workload.executables()) {
        EXPECT_EQ(rangedExecutable.index(), rangedExecutableCount);
        ++rangedExecutableCount;
    }
    EXPECT_EQ(rangedExecutableCount, workload.executableCount());

    const auto executable = workload.executable(0);
    EXPECT_EQ(executable.index(), 0);
    EXPECT_EQ(executable.name(), "executable");
    EXPECT_EQ(executable.type(), ExecutableKind::Compute);
    const auto module = executable.module();
    EXPECT_EQ(module.index(), 0);
    EXPECT_EQ(module.name(), "shader");
    EXPECT_EQ(module.entryPoint(), "main");

    ASSERT_EQ(executable.interfaceDescriptorBindingCount(), 2);
    const auto inputBinding = executable.interfaceDescriptorBinding(0);
    EXPECT_EQ(inputBinding.set, 2);
    EXPECT_EQ(inputBinding.binding, 3);
    EXPECT_EQ(inputBinding.resourceIndex, 0);
    EXPECT_EQ(inputBinding.access, ResourceAccess::Read);
    EXPECT_EQ(inputBinding.kind, ResourceKind::StorageBuffer);
    EXPECT_EQ(inputBinding.descriptorType, vk::DescriptorType::eStorageBuffer);

    const auto outputBinding = executable.interfaceDescriptorBinding(1);
    EXPECT_EQ(outputBinding.binding, 5);
    EXPECT_EQ(outputBinding.resourceIndex, 1);
    EXPECT_EQ(outputBinding.access, ResourceAccess::Write);

    const auto input = workload.resource(0);
    EXPECT_EQ(input.access(), ResourceAccess::Read);
    EXPECT_EQ(input.requirements().kind(), ResourceKind::StorageBuffer);
    EXPECT_EQ(input.requirements().descriptorType(), vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(input.requirements().format(), vk::Format::eR32Sfloat);
    EXPECT_EQ(input.requirements().byteSize(), 16);
    EXPECT_FALSE(input.requirements().participatesInAliasing());

    const auto output = workload.resource(1);
    EXPECT_EQ(output.access(), ResourceAccess::Write);
    EXPECT_EQ(output.requirements().kind(), ResourceKind::StorageBuffer);
    EXPECT_EQ(output.requirements().format(), vk::Format::eR32Sint);
    EXPECT_EQ(output.requirements().elementCount(), 8);
}

TEST(VgfWorkload, DecodesFileBackedImageSamplerAndHidesConstants) {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 1, 1});
    const std::array<int32_t, 4> constantData = {1, 2, 3, 4};
    const auto data = writeVgf([&](Encoder &encoder) {
        const auto module = encoder.AddModule(ModuleType::GRAPH, "graph", "main", code);
        const auto image = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_FORMAT_R8G8B8A8_UNORM,
                                                    {1, 3, 5, 4}, {});
        encoder.AddSamplerConfig(image, VK_FILTER_LINEAR, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                 VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
        const auto constantResource = encoder.AddConstantResource(VK_FORMAT_R32_SINT, {4}, {});
        const auto constant =
            encoder.AddConstant(constantResource, constantData.data(), constantData.size() * sizeof(int32_t), 1);
        const auto imageBinding = encoder.AddBindingSlot(0, image);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({imageBinding}, 0);
        const std::vector<GraphConstantBindingRef> constantBindings = {
            {constant.reference, constant},
        };
        encoder.AddSegmentInfo(module, "graph_segment", {descriptorSet}, {imageBinding}, {}, constantBindings);
    });

    const TempFolder tempFolder("mlworkloadlib_vgf_workload");
    const auto path = tempFolder.relative("file_test.vgf");
    {
        std::ofstream file(path, std::ios::binary);
        file.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    auto workload = Workload::fromVGF(path);

    ASSERT_EQ(workload.executableCount(), 1);
    ASSERT_EQ(workload.resourceCount(), 1);
    const auto executable = workload.executable(0);
    EXPECT_EQ(executable.type(), ExecutableKind::Graph);
    EXPECT_EQ(executable.module().name(), "graph");

    ASSERT_EQ(executable.interfaceDescriptorBindingCount(), 1);
    const auto imageBinding = executable.interfaceDescriptorBinding(0);
    EXPECT_EQ(imageBinding.set, 0);
    EXPECT_EQ(imageBinding.binding, 0);
    EXPECT_EQ(imageBinding.kind, ResourceKind::Image);
    EXPECT_EQ(imageBinding.descriptorType, vk::DescriptorType::eCombinedImageSampler);

    const auto image = workload.resource(0);
    EXPECT_EQ(image.requirements().kind(), ResourceKind::Image);
    EXPECT_EQ(image.requirements().format(), vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(image.requirements().asImage().extent(), vk::Extent3D(5, 3, 1));
    EXPECT_TRUE(image.requirements().asImage().isSampled());
    EXPECT_FALSE(image.requirements().asImage().isStorage());
    EXPECT_TRUE(image.requirements().asImage().hasRuntimeSampler());
    EXPECT_FALSE(image.requirements().asImage().requiresSamplerBinding());
}

TEST(VgfWorkload, DecodesImageWithoutSamplerConfigRequiresSamplerBinding) {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 0, 1});
    const auto data = writeVgf([&](Encoder &encoder) {
        const auto module = encoder.AddModule(ModuleType::COMPUTE, "shader", "main", code);
        const auto image = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_FORMAT_R8G8B8A8_SNORM,
                                                    {1, 2, 2, 4}, {});
        const auto imageBinding = encoder.AddBindingSlot(3, image);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({imageBinding}, 0);
        encoder.AddSegmentInfo(module, "executable", {descriptorSet}, {imageBinding}, {}, noGraphConstants, {1, 1, 1});
    });

    auto workload = Workload::fromVGF(data.data(), data.size());

    ASSERT_EQ(workload.resourceCount(), 1);
    const auto image = workload.resource(0);
    EXPECT_TRUE(image.requirements().asImage().isSampled());
    EXPECT_FALSE(image.requirements().asImage().hasRuntimeSampler());
    EXPECT_TRUE(image.requirements().asImage().requiresSamplerBinding());
}

TEST(VgfWorkload, DecodesPublicResourceAliasRequirements) {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 0, 1});
    const auto data = writeVgf([&](Encoder &encoder) {
        constexpr uint32_t aliasGroup = 17;
        const auto module = encoder.AddModule(ModuleType::COMPUTE, "shader", "main", code);
        const auto input = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SFLOAT, {4}, {4});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SFLOAT, {4}, {4}, aliasGroup);
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, outputBinding}, 0);
        encoder.AddSegmentInfo(module, "executable", {descriptorSet}, {inputBinding}, {outputBinding}, noGraphConstants,
                               {1, 1, 1});
    });

    auto workload = Workload::fromVGF(data.data(), data.size());

    ASSERT_EQ(workload.resourceCount(), 2);
    EXPECT_FALSE(workload.resource(0).requirements().participatesInAliasing());
    EXPECT_FALSE(workload.resource(0).requirements().requiresBoundMemoryInfo());
    EXPECT_TRUE(workload.resource(1).requirements().participatesInAliasing());
    EXPECT_TRUE(workload.resource(1).requirements().requiresBoundMemoryInfo());
}

/*******************************************************************************
 * Validation coverage
 *******************************************************************************/

TEST(VgfWorkload, RejectsInvalidImageMetadata) {
    const std::vector<std::vector<int64_t>> invalidShapes = {
        {3, 5},                                                                    // Invalid rank.
        {3, 0, 1},                                                                 // Invalid 3D extent.
        {2, 3, 5, 4},                                                              // Non-unit NHWC batch.
        {1, 3, 5, 3},                                                              // Invalid channel count.
        {1, 0, 5, 4},                                                              // Zero dimension.
        {1, 3, -1, 4},                                                             // Negative dimension.
        {1, 3, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1, 4}, // Dimension overflow.
    };

    for (const auto &shape : invalidShapes) {
        SCOPED_TRACE(testing::PrintToString(shape));
        const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 0, 1});
        const auto data = writeVgf([&](Encoder &encoder) {
            const auto module = encoder.AddModule(ModuleType::COMPUTE, "shader", "main", code);
            const auto image = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        VK_FORMAT_R8G8B8A8_SNORM, shape, {});
            const auto imageBinding = encoder.AddBindingSlot(3, image);
            const auto descriptorSet = encoder.AddDescriptorSetInfo({imageBinding}, 0);
            encoder.AddSegmentInfo(module, "executable", {descriptorSet}, {imageBinding}, {}, noGraphConstants,
                                   {1, 1, 1});
        });

        auto workload = Workload::fromVGF(data.data(), data.size());

        EXPECT_THROW((void)workload.resource(0).requirements().asImage().extent(), std::runtime_error);
    }
}
