/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/context.hpp"

#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mlworkloadlib::samples {

inline bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const char *name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
        return std::string_view(extension.extensionName.data()) == name;
    });
}

inline uint32_t findExecutionQueueFamily(const vk::raii::PhysicalDevice &physicalDevice) {
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i) {
        const auto requiredFlags = vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eDataGraphARM;
        if ((queueFamilies[i].queueFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

class ApplicationVulkanContext {
  public:
    ApplicationVulkanContext() {
        const vk::ApplicationInfo applicationInfo("mlworkloadlib-sample", 1, nullptr, 0, VK_API_VERSION_1_3);
        std::vector<const char *> instanceExtensions;
        vk::InstanceCreateFlags instanceFlags;
        const auto availableInstanceExtensions = raiiContext_.enumerateInstanceExtensionProperties();
        if (hasExtension(availableInstanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceFlags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        }
        instance_ =
            vk::raii::Instance(raiiContext_, vk::InstanceCreateInfo(instanceFlags, &applicationInfo, {}, {},
                                                                    static_cast<uint32_t>(instanceExtensions.size()),
                                                                    instanceExtensions.data()));

        for (auto &candidate : vk::raii::PhysicalDevices(instance_)) {
            const auto extensions = candidate.enumerateDeviceExtensionProperties();
            if (!hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME)) {
                continue;
            }
            const auto queueFamilyIndex = findExecutionQueueFamily(candidate);
            if (queueFamilyIndex != std::numeric_limits<uint32_t>::max()) {
                physicalDevice_ = candidate;
                queueFamilyIndex_ = queueFamilyIndex;
                break;
            }
        }
        if (queueFamilyIndex_ == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("No Vulkan device supports the sample context requirements");
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex_, 1, &queuePriority);

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.shaderInt16 = true;
        deviceFeatures.shaderInt64 = true;

        vk::PhysicalDeviceVulkan12Features vulkan12Features;
        vulkan12Features.storageBuffer8BitAccess = true;
        vulkan12Features.shaderInt8 = true;
        vulkan12Features.vulkanMemoryModel = true;

        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vulkan13Features.synchronization2 = true;
        vulkan13Features.pipelineCreationCacheControl = true;
        vulkan13Features.pNext = &vulkan12Features;

        vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5Features;
        maintenance5Features.maintenance5 = true;
        maintenance5Features.pNext = &vulkan13Features;

        vk::PhysicalDeviceTensorFeaturesARM tensorFeatures;
        tensorFeatures.tensors = true;
        tensorFeatures.shaderTensorAccess = true;
        tensorFeatures.tensorNonPacked = true;
        tensorFeatures.pNext = &maintenance5Features;

        vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeatures;
        dataGraphFeatures.dataGraph = true;
        dataGraphFeatures.dataGraphShaderModule = true;
        dataGraphFeatures.dataGraphSpecializationConstants = true;
        dataGraphFeatures.pNext = &tensorFeatures;

        vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures;
        void *featureChain = &dataGraphFeatures;
        const auto extensions = physicalDevice_.enumerateDeviceExtensionProperties();
        std::vector<const char *> deviceExtensions = {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                                      VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
                                                      VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_TENSORS_EXTENSION_NAME};
        if (hasExtension(extensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) {
            replicatedCompositesFeatures.shaderReplicatedComposites = true;
            replicatedCompositesFeatures.pNext = featureChain;
            featureChain = &replicatedCompositesFeatures;
            deviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        }
        if (hasExtension(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }

        device_ = vk::raii::Device(
            physicalDevice_,
            {vk::DeviceCreateFlags(), queueCreateInfo, {}, deviceExtensions, &deviceFeatures, featureChain});
        queue_ = device_.getQueue(queueFamilyIndex_, 0);
    }

    ContextView view() const { return {instance_, physicalDevice_, device_, queueFamilyIndex_, queue_}; }

  private:
    vk::raii::Context raiiContext_;
    vk::raii::Instance instance_{nullptr};
    vk::raii::PhysicalDevice physicalDevice_{nullptr};
    vk::raii::Device device_{nullptr};
    vk::raii::Queue queue_{nullptr};
    uint32_t queueFamilyIndex_ = std::numeric_limits<uint32_t>::max();
};

} // namespace mlworkloadlib::samples
