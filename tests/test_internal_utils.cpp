/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/utils.hpp"
#include "internal/workload_impl.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::detail;

Resource makeStridedBufferResource(std::vector<int64_t> shape, std::vector<int64_t> stride) {
    Resource resource;
    resource.descriptorType = vk::DescriptorType::eStorageBuffer;
    resource.format = vk::Format::eR32Sint;
    resource.shape = std::move(shape);
    resource.stride = std::move(stride);
    resource.metadata = Resource::BufferMetadata{0, vk::BufferUsageFlagBits::eStorageBuffer};
    return resource;
}

TEST(InternalUtils, RejectsElementCountOverflow) {
    const std::vector<int64_t> shape = {std::numeric_limits<int64_t>::max(), 3};

    EXPECT_THROW((void)elementCount(shape), std::runtime_error);
}

TEST(InternalUtils, ComputesStridedResourceByteSizeWithCheckedArithmetic) {
    const auto resource = makeStridedBufferResource({4}, {static_cast<int64_t>(sizeof(int32_t))});

    EXPECT_EQ(
        storageBufferByteSize(bufferMetadata(resource).byteSize, resource.format, resource.shape, resource.stride),
        4 * sizeof(int32_t));
}

TEST(InternalUtils, RejectsInvalidStridedResourceShape) {
    const auto resource = makeStridedBufferResource({0}, {static_cast<int64_t>(sizeof(int32_t))});

    EXPECT_THROW((void)storageBufferByteSize(bufferMetadata(resource).byteSize, resource.format, resource.shape,
                                             resource.stride),
                 std::runtime_error);
}

TEST(InternalUtils, RejectsNegativeResourceStride) {
    const auto resource = makeStridedBufferResource({4}, {-1});

    EXPECT_THROW((void)storageBufferByteSize(bufferMetadata(resource).byteSize, resource.format, resource.shape,
                                             resource.stride),
                 std::runtime_error);
}

TEST(InternalUtils, RejectsResourceByteSizeOverflow) {
    const auto resource = makeStridedBufferResource({std::numeric_limits<int64_t>::max()}, {3});

    EXPECT_THROW((void)storageBufferByteSize(bufferMetadata(resource).byteSize, resource.format, resource.shape,
                                             resource.stride),
                 std::runtime_error);
}

TEST(InternalUtils, SplitsDescriptorBindingsWithinDeviceLimit) {
    const std::vector<DescriptorBinding> bindings = {{0, 1, 0, ResourceAccess::Read}};

    const auto sets = splitBindingsBySet(bindings, 2);

    ASSERT_EQ(sets.size(), 2);
    ASSERT_EQ(sets[0].size(), 0);
    ASSERT_EQ(sets[1].size(), 1);
    EXPECT_EQ(sets[1][0].set, 1);
}

TEST(InternalUtils, RejectsDescriptorSetBeyondDeviceLimit) {
    const std::vector<DescriptorBinding> bindings = {{0, 2, 0, ResourceAccess::Read}};

    EXPECT_THROW((void)splitBindingsBySet(bindings, 2), std::runtime_error);
}

} // namespace
