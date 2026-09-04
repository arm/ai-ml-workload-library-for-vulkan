/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "internal/utils.hpp"

#include "mlworkloadlib/binding_types.hpp"

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlworkloadlib::test {

/*******************************************************************************
 * Mapped device memory helpers
 *******************************************************************************/

class MappedDeviceMemory {
  public:
    MappedDeviceMemory(const vk::raii::Device &device, BoundMemoryInfo memory) : device_(device), memory_(memory) {
        const auto mapResult = static_cast<vk::Result>(device_.getDispatcher()->vkMapMemory(
            static_cast<VkDevice>(*device_), static_cast<VkDeviceMemory>(memory_.memory), memory_.offset, memory_.size,
            0, &data_));
        if (mapResult != vk::Result::eSuccess) {
            throw std::runtime_error("vkMapMemory failed");
        }
    }

    ~MappedDeviceMemory() {
        device_.getDispatcher()->vkUnmapMemory(static_cast<VkDevice>(*device_),
                                               static_cast<VkDeviceMemory>(memory_.memory));
    }

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

inline void validateMappedMemoryAccess(BoundMemoryInfo memory, vk::DeviceSize byteSize, const char *errorMessage) {
    if (memory.memory == nullptr || byteSize > memory.size) {
        throw std::runtime_error(errorMessage);
    }
}

template <typename T>
inline void writeMappedMemory(const vk::raii::Device &device, BoundMemoryInfo memory, const std::vector<T> &values) {
    const auto byteSize = static_cast<vk::DeviceSize>(values.size() * sizeof(T));
    validateMappedMemoryAccess(memory, byteSize, "Mapped memory write exceeds allocation size");
    const MappedDeviceMemory mappedMemory(device, memory);
    std::memset(mappedMemory.data(), 0, static_cast<std::size_t>(memory.size));
    std::copy(values.begin(), values.end(), static_cast<T *>(mappedMemory.data()));
}

template <typename T>
inline std::vector<T> readMappedMemory(const vk::raii::Device &device, BoundMemoryInfo memory, std::size_t elements) {
    const auto byteSize = static_cast<vk::DeviceSize>(elements * sizeof(T));
    validateMappedMemoryAccess(memory, byteSize, "Mapped memory read exceeds allocation size");
    const MappedDeviceMemory mappedMemory(device, memory);
    const auto *begin = static_cast<const T *>(mappedMemory.data());
    return {begin, begin + elements};
}

/*******************************************************************************
 * Vulkan test resources
 *******************************************************************************/

struct Tensor {
    Tensor(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, vk::Format format,
           std::vector<int64_t> shapeIn)
        : shape(std::move(shapeIn)) {
        const vk::TensorDescriptionARM description(vk::TensorTilingARM::eLinear, format,
                                                   static_cast<uint32_t>(this->shape.size()), this->shape.data(),
                                                   nullptr, vk::TensorUsageFlagBitsARM::eDataGraph);
        const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);
        tensor = vk::raii::TensorARM(device, createInfo);

        const auto memoryRequirements =
            device.getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*tensor));
        memorySize = memoryRequirements.memoryRequirements.size;
        const auto memoryType = detail::vulkan_helpers::findMemoryType(
            physicalDevice, memoryRequirements.memoryRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        device.bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*tensor, *memory, 0));
    }

    std::size_t numElements() const { return Tensor::numElements(shape); }

    static std::size_t numElements(const std::vector<int64_t> &shape) {
        return static_cast<std::size_t>(detail::utils::elementCount(shape));
    }

    void fill(int8_t value, std::size_t elements) const {
        void *data = memory.mapMemory(0, memorySize);
        std::memset(data, 0, static_cast<std::size_t>(memorySize));
        std::fill_n(static_cast<int8_t *>(data), elements, value);
        memory.unmapMemory();
    }

    void write(const std::vector<int8_t> &values) const {
        void *data = memory.mapMemory(0, memorySize);
        std::memset(data, 0, static_cast<std::size_t>(memorySize));
        std::copy(values.begin(), values.end(), static_cast<int8_t *>(data));
        memory.unmapMemory();
    }

    std::vector<int8_t> read(std::size_t elements) const {
        const void *data = memory.mapMemory(0, memorySize);
        const auto *begin = static_cast<const int8_t *>(data);
        std::vector<int8_t> result(begin, begin + elements);
        memory.unmapMemory();
        return result;
    }

    std::vector<int64_t> shape;
    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::TensorARM tensor{nullptr};
    vk::DeviceSize memorySize = 0;
};

struct Buffer {
    Buffer(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, vk::DeviceSize size)
        : buffer(device, vk::BufferCreateInfo({}, size, vk::BufferUsageFlagBits::eStorageBuffer)) {
        const auto memoryRequirements = buffer.getMemoryRequirements();
        memorySize = memoryRequirements.size;
        const auto memoryType = detail::vulkan_helpers::findMemoryType(
            physicalDevice, memoryRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        buffer.bindMemory(*memory, 0);
    }

    void write(const std::vector<int32_t> &values) const {
        void *data = memory.mapMemory(0, memorySize);
        std::memset(data, 0, static_cast<std::size_t>(memorySize));
        std::copy(values.begin(), values.end(), static_cast<int32_t *>(data));
        memory.unmapMemory();
    }

    std::vector<int32_t> read(std::size_t elements) const {
        const void *data = memory.mapMemory(0, memorySize);
        const auto *begin = static_cast<const int32_t *>(data);
        std::vector<int32_t> result(begin, begin + elements);
        memory.unmapMemory();
        return result;
    }

    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::Buffer buffer{nullptr};
    vk::DeviceSize memorySize = 0;
};

struct Image {
    Image(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device,
          vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, vk::Extent3D extent = {2, 2, 1})
        : image(device, vk::ImageCreateInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Snorm, extent, 1, 1,
                                            vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage,
                                            vk::SharingMode::eExclusive)) {
        const auto memoryRequirements = image.getMemoryRequirements();
        memorySize = memoryRequirements.size;
        const auto memoryType = detail::vulkan_helpers::findMemoryType(
            physicalDevice, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        image.bindMemory(*memory, 0);

        const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        imageView = vk::raii::ImageView(
            device, vk::ImageViewCreateInfo({}, *image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Snorm, {}, range));
    }

    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::Image image{nullptr};
    vk::raii::ImageView imageView{nullptr};
    vk::DeviceSize memorySize = 0;
};

} // namespace mlworkloadlib::test
