/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib_utils/application_context.hpp"

#include "internal/utils.hpp"

#include <vulkan/vulkan_beta.h>

#include <stdexcept>
#include <utility>

namespace mlsdk::workloadlib::utils {

ApplicationContext::ApplicationContext(const RuntimeContextDeviceRequirements &requirements) {
    if (!initialize(requirements)) {
        throw std::runtime_error("No Vulkan device supports the application context requirements");
    }
}

bool ApplicationContext::initialize(const RuntimeContextDeviceRequirements &requirements) {
    instance = detail::createVulkanInstance(raiiContext, "mlworkloadlib");

    const auto requiredDeviceExtensions = detail::requiredWorkloadDeviceExtensions(requirements);
    const auto requiredQueueFlags = detail::requiredWorkloadQueueFlags(requirements);
    for (auto &candidate : vk::raii::PhysicalDevices(instance)) {
        const auto extensions = candidate.enumerateDeviceExtensionProperties();
        if (!detail::hasExtensions(extensions, requiredDeviceExtensions)) {
            continue;
        }
        const auto candidateQueueFamilyIndex =
            detail::findQueueFamilyIndex(candidate.getQueueFamilyProperties(), requiredQueueFlags);
        if (!candidateQueueFamilyIndex.has_value()) {
            continue;
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, *candidateQueueFamilyIndex, 1, &queuePriority);
        try {
            const detail::RuntimeDeviceConfiguration deviceConfiguration(extensions, requiredDeviceExtensions,
                                                                         requirements);
            auto candidateDevice = vk::raii::Device(candidate, {vk::DeviceCreateFlags(),
                                                                queueCreateInfo,
                                                                {},
                                                                deviceConfiguration.extensions(),
                                                                &deviceConfiguration.features(),
                                                                deviceConfiguration.featureChain()});
            auto candidateQueue = candidateDevice.getQueue(*candidateQueueFamilyIndex, 0);
            physicalDevice = candidate;
            device = std::move(candidateDevice);
            queue = std::move(candidateQueue);
            queueFamilyIndex = *candidateQueueFamilyIndex;
            return true;
        } catch (const vk::SystemError &error) {
            if (error.code() != vk::make_error_code(vk::Result::eErrorFeatureNotPresent) &&
                error.code() != vk::make_error_code(vk::Result::eErrorExtensionNotPresent)) {
                throw;
            }
        }
    }
    return false;
}

const vk::raii::Instance &ApplicationContext::vulkanInstance() const noexcept { return instance; }

const vk::raii::PhysicalDevice &ApplicationContext::vulkanPhysicalDevice() const noexcept { return physicalDevice; }

const vk::raii::Device &ApplicationContext::vulkanDevice() const noexcept { return device; }

const vk::raii::Queue &ApplicationContext::vulkanQueue() const noexcept { return queue; }

uint32_t ApplicationContext::vulkanQueueFamilyIndex() const noexcept { return queueFamilyIndex; }

} // namespace mlsdk::workloadlib::utils
