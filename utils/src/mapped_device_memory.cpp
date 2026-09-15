/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib_utils/mapped_device_memory.hpp"

#include <vulkan/vulkan_core.h>

#include <stdexcept>

namespace mlsdk::workloadlib::utils {

MappedDeviceMemory::MappedDeviceMemory(const vk::raii::Device &device, BoundMemoryInfo memory)
    : device_(device), memory_(memory) {
    if (memory_.memory == nullptr) {
        throw std::runtime_error("Cannot map a null memory allocation");
    }
    const auto result = static_cast<vk::Result>(device_.getDispatcher()->vkMapMemory(
        static_cast<VkDevice>(*device_), static_cast<VkDeviceMemory>(memory_.memory), memory_.offset, memory_.size, 0,
        &data_));
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("vkMapMemory failed");
    }
}

MappedDeviceMemory::~MappedDeviceMemory() {
    device_.getDispatcher()->vkUnmapMemory(static_cast<VkDevice>(*device_),
                                           static_cast<VkDeviceMemory>(memory_.memory));
}

} // namespace mlsdk::workloadlib::utils
