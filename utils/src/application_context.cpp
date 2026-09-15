/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib_utils/application_context.hpp"

#include "internal/utils.hpp"

#include <vulkan/vulkan_beta.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlsdk::workloadlib::utils {

ApplicationContext::ApplicationContext(std::string_view applicationName) {
    if (!initialize(applicationName)) {
        throw std::runtime_error("No Vulkan device supports the application context requirements");
    }
}

bool ApplicationContext::initialize(std::string_view applicationName) {
    const std::string applicationNameStorage(applicationName);
    const vk::ApplicationInfo applicationInfo(applicationNameStorage.c_str(), 1, nullptr, 0, VK_API_VERSION_1_3);
    std::vector<const char *> instanceExtensions;
    vk::InstanceCreateFlags instanceFlags;
    const auto availableInstanceExtensions = raiiContext.enumerateInstanceExtensionProperties();
    if (detail::hasExtension(availableInstanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instanceFlags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
    instance = vk::raii::Instance(raiiContext, vk::InstanceCreateInfo(instanceFlags, &applicationInfo, {}, {},
                                                                      static_cast<uint32_t>(instanceExtensions.size()),
                                                                      instanceExtensions.data()));

    const auto requiredDeviceExtensions = detail::requiredWorkloadDeviceExtensions();
    for (auto &candidate : vk::raii::PhysicalDevices(instance)) {
        const auto extensions = candidate.enumerateDeviceExtensionProperties();
        if (!detail::hasExtensions(extensions, requiredDeviceExtensions)) {
            continue;
        }
        const auto requiredQueueFlags = vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eDataGraphARM;
        const auto candidateQueueFamilyIndex = detail::findQueueFamilyIndex(candidate, requiredQueueFlags);
        if (candidateQueueFamilyIndex.has_value()) {
            physicalDevice = candidate;
            queueFamilyIndex = *candidateQueueFamilyIndex;
            break;
        }
    }
    if (queueFamilyIndex == std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const float queuePriority = 1.0F;
    const vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);
    const auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
    const detail::RuntimeDeviceConfiguration deviceConfiguration(extensions, requiredDeviceExtensions);
    device = vk::raii::Device(physicalDevice, {vk::DeviceCreateFlags(),
                                               queueCreateInfo,
                                               {},
                                               deviceConfiguration.extensions(),
                                               &deviceConfiguration.features(),
                                               deviceConfiguration.featureChain()});
    queue = device.getQueue(queueFamilyIndex, 0);
    return true;
}

const vk::raii::Instance &ApplicationContext::vulkanInstance() const noexcept { return instance; }

const vk::raii::PhysicalDevice &ApplicationContext::vulkanPhysicalDevice() const noexcept { return physicalDevice; }

const vk::raii::Device &ApplicationContext::vulkanDevice() const noexcept { return device; }

const vk::raii::Queue &ApplicationContext::vulkanQueue() const noexcept { return queue; }

uint32_t ApplicationContext::vulkanQueueFamilyIndex() const noexcept { return queueFamilyIndex; }

} // namespace mlsdk::workloadlib::utils
