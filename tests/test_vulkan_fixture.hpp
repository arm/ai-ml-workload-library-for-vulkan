/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/context.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mlworkloadlib::test {

/*******************************************************************************
 * Vulkan fixture
 *******************************************************************************/

inline bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const char *name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
        return std::string_view(extension.extensionName.data()) == name;
    });
}

inline uint32_t findDataGraphQueueFamily(const vk::raii::PhysicalDevice &physicalDevice) {
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i) {
        const auto requiredFlags = vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eCompute;
        if ((queueFamilies[i].queueFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    return UINT32_MAX;
}

class RuntimeSessionExecutionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const vk::ApplicationInfo applicationInfo("mlworkloadlib-test", 1, nullptr, 0, VK_API_VERSION_1_3);
        std::vector<const char *> instanceExtensions;
        vk::InstanceCreateFlags instanceFlags;
        const auto availableInstanceExtensions = raiiContext.enumerateInstanceExtensionProperties();
        if (hasExtension(availableInstanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceFlags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        }
        instance =
            vk::raii::Instance(raiiContext, vk::InstanceCreateInfo(instanceFlags, &applicationInfo, {}, {},
                                                                   static_cast<uint32_t>(instanceExtensions.size()),
                                                                   instanceExtensions.data()));

        for (auto &candidate : vk::raii::PhysicalDevices(instance)) {
            const auto extensions = candidate.enumerateDeviceExtensionProperties();
            if (!hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME)) {
                continue;
            }
            const auto candidateQueueFamilyIndex = findDataGraphQueueFamily(candidate);
            if (candidateQueueFamilyIndex != UINT32_MAX) {
                physicalDevice = candidate;
                queueFamilyIndex = candidateQueueFamilyIndex;
                break;
            }
        }
        if (queueFamilyIndex == UINT32_MAX) {
            GTEST_SKIP() << "No Vulkan device with required data graph extensions and compute queue support";
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.shaderInt16 = true;
        deviceFeatures.shaderInt64 = true;

        vulkan12Features.storageBuffer8BitAccess = true;
        vulkan12Features.shaderInt8 = true;
        vulkan12Features.vulkanMemoryModel = true;

        vulkan13Features.synchronization2 = true;
        vulkan13Features.pipelineCreationCacheControl = true;
        vulkan13Features.pNext = &vulkan12Features;

        maintenance5Features.maintenance5 = true;
        maintenance5Features.pNext = &vulkan13Features;

        tensorFeatures.tensors = true;
        tensorFeatures.shaderTensorAccess = true;
        tensorFeatures.tensorNonPacked = true;
        tensorFeatures.pNext = &maintenance5Features;

        dataGraphFeatures.dataGraph = true;
        dataGraphFeatures.dataGraphShaderModule = true;
        dataGraphFeatures.dataGraphSpecializationConstants = true;
        dataGraphFeatures.pNext = &tensorFeatures;

        void *featureChain = &dataGraphFeatures;
        const auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
        std::vector<const char *> deviceExtensions = {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                                      VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
                                                      VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_TENSORS_EXTENSION_NAME};
        if (hasExtension(extensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) {
            replicatedCompositesFeatures.shaderReplicatedComposites = true;
            replicatedCompositesFeatures.pNext = featureChain;
            featureChain = &replicatedCompositesFeatures;
            deviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        }
        // Enable extension if available, as it is required for some platforms (e.g. MoltenVK on Darwin).
        if (hasExtension(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
        device = vk::raii::Device(
            physicalDevice,
            {vk::DeviceCreateFlags(), queueCreateInfo, {}, deviceExtensions, &deviceFeatures, featureChain});
        queue = device.getQueue(queueFamilyIndex, 0);
    }

    Context wrappedContext() { return Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue}); }

    vk::raii::Context raiiContext;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue queue{nullptr};
    uint32_t queueFamilyIndex = UINT32_MAX;

    vk::PhysicalDeviceVulkan12Features vulkan12Features;
    vk::PhysicalDeviceVulkan13Features vulkan13Features;
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5Features;
    vk::PhysicalDeviceTensorFeaturesARM tensorFeatures;
    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeatures;
    vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures;
};

} // namespace mlworkloadlib::test
