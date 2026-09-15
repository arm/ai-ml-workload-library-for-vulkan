/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_types.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mlsdk::workloadlib::utils {

// Scoped mapping of a host-visible Vulkan device-memory range. The device and
// memory allocation must remain valid for the lifetime of this object.
class MappedDeviceMemory {
  public:
    MappedDeviceMemory(const vk::raii::Device &device, BoundMemoryInfo memory);
    ~MappedDeviceMemory();

    MappedDeviceMemory(const MappedDeviceMemory &) = delete;
    MappedDeviceMemory &operator=(const MappedDeviceMemory &) = delete;
    MappedDeviceMemory(MappedDeviceMemory &&) = delete;
    MappedDeviceMemory &operator=(MappedDeviceMemory &&) = delete;

    void *data() const noexcept { return data_; }

  private:
    const vk::raii::Device &device_;
    BoundMemoryInfo memory_;
    void *data_ = nullptr;
};

inline void clearDeviceMemory(const vk::raii::Device &device, BoundMemoryInfo memory) {
    const MappedDeviceMemory mappedMemory(device, memory);
    std::memset(mappedMemory.data(), 0, static_cast<std::size_t>(memory.size));
}

template <typename T>
void writeDeviceMemory(const vk::raii::Device &device, BoundMemoryInfo memory, const std::vector<T> &values) {
    static_assert(std::is_trivially_copyable_v<T>, "Mapped memory values must be trivially copyable");
    if (values.size() > memory.size / sizeof(T)) {
        throw std::runtime_error("Mapped memory write exceeds allocation size");
    }
    const auto byteSize = static_cast<vk::DeviceSize>(values.size() * sizeof(T));
    const MappedDeviceMemory mappedMemory(device, memory);
    if (byteSize != 0) {
        std::memcpy(mappedMemory.data(), values.data(), static_cast<std::size_t>(byteSize));
    }
}

template <typename T>
std::vector<T> readDeviceMemory(const vk::raii::Device &device, BoundMemoryInfo memory, std::size_t elementCount) {
    static_assert(std::is_trivially_copyable_v<T>, "Mapped memory values must be trivially copyable");
    if (elementCount > memory.size / sizeof(T)) {
        throw std::runtime_error("Mapped memory read exceeds allocation size");
    }
    const auto byteSize = static_cast<vk::DeviceSize>(elementCount * sizeof(T));
    const MappedDeviceMemory mappedMemory(device, memory);
    std::vector<T> values(elementCount);
    if (byteSize != 0) {
        std::memcpy(values.data(), mappedMemory.data(), static_cast<std::size_t>(byteSize));
    }
    return values;
}

} // namespace mlsdk::workloadlib::utils
