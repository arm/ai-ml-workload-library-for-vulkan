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

TEST(InternalUtils, PreservesDefaultRuntimeDeviceRequirements) {
    const RuntimeContextDeviceRequirements requirements;
    const auto extensions = requiredWorkloadDeviceExtensions(requirements);
    EXPECT_TRUE(hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME));
    EXPECT_TRUE(hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME));
    EXPECT_TRUE(hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME));
    EXPECT_TRUE(hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME));
    EXPECT_EQ(requiredWorkloadQueueFlags(requirements), vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eDataGraphARM);

    const RuntimeDeviceConfiguration configuration({}, extensions, requirements);
    EXPECT_TRUE(configuration.features().shaderInt16);
    EXPECT_TRUE(configuration.features().shaderInt64);
    const auto *graphFeatures =
        static_cast<const vk::PhysicalDeviceDataGraphFeaturesARM *>(configuration.featureChain());
    EXPECT_TRUE(graphFeatures->dataGraph);
    EXPECT_TRUE(graphFeatures->dataGraphShaderModule);
    EXPECT_TRUE(graphFeatures->dataGraphSpecializationConstants);
    const auto *tensorFeatures = static_cast<const vk::PhysicalDeviceTensorFeaturesARM *>(graphFeatures->pNext);
    EXPECT_TRUE(tensorFeatures->tensors);
}

TEST(InternalUtils, SelectsComputeOnlyRuntimeDeviceCapability) {
    RuntimeContextDeviceRequirements requirements;
    requirements.capabilities = RuntimeDeviceCapability::Compute;

    const auto extensions = requiredWorkloadDeviceExtensions(requirements);
    EXPECT_FALSE(hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME));
    EXPECT_FALSE(hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME));
    EXPECT_FALSE(hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME));
    EXPECT_FALSE(hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME));
    EXPECT_EQ(requiredWorkloadQueueFlags(requirements), vk::QueueFlagBits::eCompute);

    const RuntimeDeviceConfiguration configuration({}, extensions, requirements);
    EXPECT_TRUE(configuration.features().shaderInt16);
    const auto *vulkan13Features =
        static_cast<const vk::PhysicalDeviceVulkan13Features *>(configuration.featureChain());
    EXPECT_EQ(vulkan13Features->sType, vk::StructureType::ePhysicalDeviceVulkan13Features);
    EXPECT_TRUE(vulkan13Features->synchronization2);
}

TEST(InternalUtils, SelectsDataGraphOnlyRuntimeDeviceCapability) {
    RuntimeContextDeviceRequirements requirements;
    requirements.capabilities = RuntimeDeviceCapability::DataGraph;

    const auto extensions = requiredWorkloadDeviceExtensions(requirements);
    EXPECT_TRUE(hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME));
    EXPECT_TRUE(hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME));
    EXPECT_TRUE(hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME));
    EXPECT_FALSE(hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME));
    EXPECT_EQ(requiredWorkloadQueueFlags(requirements), vk::QueueFlagBits::eDataGraphARM);

    const RuntimeDeviceConfiguration configuration({}, extensions, requirements);
    const auto *graphFeatures =
        static_cast<const vk::PhysicalDeviceDataGraphFeaturesARM *>(configuration.featureChain());
    EXPECT_TRUE(graphFeatures->dataGraph);
    EXPECT_FALSE(graphFeatures->dataGraphShaderModule);
    EXPECT_FALSE(graphFeatures->dataGraphSpecializationConstants);
    EXPECT_FALSE(configuration.features().shaderInt16);
}

TEST(InternalUtils, ShaderModulesIncludeDataGraphCapability) {
    RuntimeContextDeviceRequirements requirements;
    requirements.capabilities = RuntimeDeviceCapability::DataGraphShaderModules;
    const auto extensions = requiredWorkloadDeviceExtensions(requirements);
    EXPECT_TRUE(hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME));
    EXPECT_EQ(requiredWorkloadQueueFlags(requirements), vk::QueueFlagBits::eDataGraphARM);

    const RuntimeDeviceConfiguration configuration({}, extensions, requirements);
    const auto *graphFeatures =
        static_cast<const vk::PhysicalDeviceDataGraphFeaturesARM *>(configuration.featureChain());
    EXPECT_TRUE(graphFeatures->dataGraph);
    EXPECT_TRUE(graphFeatures->dataGraphShaderModule);
    EXPECT_TRUE(configuration.features().shaderInt16);
}

TEST(InternalUtils, RejectsInvalidRuntimeDeviceCapabilities) {
    RuntimeContextDeviceRequirements requirements;
    requirements.capabilities = RuntimeDeviceCapability{};
    EXPECT_THROW((void)requiredWorkloadQueueFlags(requirements), std::runtime_error);

    requirements.capabilities = static_cast<RuntimeDeviceCapability>(0x09);
    EXPECT_THROW((void)requiredWorkloadQueueFlags(requirements), std::runtime_error);

    requirements.capabilities = static_cast<RuntimeDeviceCapability>(0x10);
    EXPECT_THROW((void)requiredWorkloadQueueFlags(requirements), std::runtime_error);
}

} // namespace
