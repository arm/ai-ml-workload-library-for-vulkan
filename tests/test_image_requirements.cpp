/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/workload_builder.hpp"
#include "test_resource_requirements.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace {

using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::test;

TEST(ImageRequirements, DerivesDefaultImageSubresourceRangeFromFormat) {
    struct ImageAspectCase {
        vk::Format format;
        vk::ImageAspectFlags storageAspects;
        vk::ImageAspectFlags sampledAspects;
    };
    constexpr auto depthStencilAspects = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    constexpr std::array cases{
        ImageAspectCase{vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eColor},
        ImageAspectCase{vk::Format::eD16Unorm, vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eDepth},
        ImageAspectCase{vk::Format::eS8Uint, vk::ImageAspectFlagBits::eStencil, vk::ImageAspectFlagBits::eStencil},
        ImageAspectCase{vk::Format::eD16UnormS8Uint, depthStencilAspects, vk::ImageAspectFlagBits::eDepth},
        ImageAspectCase{vk::Format::eD24UnormS8Uint, depthStencilAspects, vk::ImageAspectFlagBits::eDepth},
        ImageAspectCase{vk::Format::eD32SfloatS8Uint, depthStencilAspects, vk::ImageAspectFlagBits::eDepth},
    };

    for (const auto &testCase : cases) {
        SCOPED_TRACE(static_cast<int32_t>(testCase.format));
        for (const bool sampled : {false, true}) {
            SCOPED_TRACE(sampled ? "sampled image" : "storage image");
            auto requirements = makeStorageImageRequirements();
            requirements.format = testCase.format;
            requirements.image.range = vk::ImageSubresourceRange{};
            if (sampled) {
                requirements.descriptorType = vk::DescriptorType::eCombinedImageSampler;
                requirements.image.usage = vk::ImageUsageFlagBits::eSampled;
                requirements.image.requiredLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            }
            detail::WorkloadBuilder builder;
            const auto imageIndex = builder.addResource("image", requirements, detail::Resource::Role::Input);
            const auto workload = builder.finish();

            const auto range = workload.resource(imageIndex).requirements().asImage().requiredSubresourceRange();
            const auto expectedAspects = sampled ? testCase.sampledAspects : testCase.storageAspects;
            EXPECT_EQ(range, vk::ImageSubresourceRange(expectedAspects, 0, 1, 0, 1));
        }
    }
}

TEST(ImageRequirements, RejectsImageSubresourceAspectsIncompatibleWithFormat) {
    struct ImageAspectCase {
        vk::Format format;
        vk::ImageAspectFlags aspects;
    };
    constexpr std::array cases{
        ImageAspectCase{vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eDepth},
        ImageAspectCase{vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor | vk::ImageAspectFlagBits::eDepth},
        ImageAspectCase{vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eMetadata},
        ImageAspectCase{vk::Format::eD16Unorm, vk::ImageAspectFlagBits::eColor},
        ImageAspectCase{vk::Format::eD16Unorm, vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil},
        ImageAspectCase{vk::Format::eS8Uint, vk::ImageAspectFlagBits::eDepth},
        ImageAspectCase{vk::Format::eD24UnormS8Uint, vk::ImageAspectFlagBits::eColor},
    };

    for (const auto &testCase : cases) {
        SCOPED_TRACE(static_cast<int32_t>(testCase.format));
        SCOPED_TRACE(static_cast<VkImageAspectFlags>(testCase.aspects));
        auto requirements = makeStorageImageRequirements();
        requirements.format = testCase.format;
        requirements.image.range = {testCase.aspects, 0, 1, 0, 1};
        detail::WorkloadBuilder builder;

        EXPECT_THROW((void)builder.addResource("image", requirements, detail::Resource::Role::Input),
                     std::runtime_error);
    }
}

TEST(ImageRequirements, RejectsSampledDepthStencilSubresourceRange) {
    for (const auto format : {vk::Format::eD16UnormS8Uint, vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint}) {
        SCOPED_TRACE(static_cast<int32_t>(format));
        auto requirements = makeStorageImageRequirements();
        requirements.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        requirements.image.usage = vk::ImageUsageFlagBits::eSampled;
        requirements.image.requiredLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        requirements.format = format;
        requirements.image.range = {vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1};
        detail::WorkloadBuilder builder;

        EXPECT_THROW((void)builder.addResource("image", requirements, detail::Resource::Role::Input),
                     std::runtime_error);
    }
}

TEST(ImageRequirements, RejectsUnknownImageFormats) {
    for (const auto format : {vk::Format::eUndefined, static_cast<vk::Format>(VK_FORMAT_MAX_ENUM)}) {
        SCOPED_TRACE(static_cast<int32_t>(format));
        auto requirements = makeStorageImageRequirements();
        requirements.format = format;
        requirements.image.range = vk::ImageSubresourceRange{};
        detail::WorkloadBuilder builder;

        EXPECT_THROW((void)builder.addResource("image", requirements, detail::Resource::Role::Input),
                     std::runtime_error);
    }
}

} // namespace
