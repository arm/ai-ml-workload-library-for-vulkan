/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/binding_types.hpp"
#include "mlworkloadlib/workload.hpp"

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <vgf/encoder.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlworkloadlib::samples {

class MappedMemory {
  public:
    MappedMemory(const vk::raii::Device &device, BoundMemoryInfo memory) : device_(device), memory_(memory) {
        const auto result = static_cast<vk::Result>(device_.getDispatcher()->vkMapMemory(
            static_cast<VkDevice>(*device_), static_cast<VkDeviceMemory>(memory_.memory), memory_.offset, memory_.size,
            0, &data_));
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("vkMapMemory failed");
        }
    }

    ~MappedMemory() {
        device_.getDispatcher()->vkUnmapMemory(static_cast<VkDevice>(*device_),
                                               static_cast<VkDeviceMemory>(memory_.memory));
    }

    MappedMemory(const MappedMemory &) = delete;
    MappedMemory &operator=(const MappedMemory &) = delete;

    void *data() const noexcept { return data_; }

  private:
    const vk::raii::Device &device_;
    BoundMemoryInfo memory_;
    void *data_ = nullptr;
};

inline void clearMemory(const vk::raii::Device &device, BoundMemoryInfo memory) {
    if (memory.memory == nullptr) {
        throw std::runtime_error("Cannot map a null memory allocation");
    }
    const MappedMemory mapped(device, memory);
    std::memset(mapped.data(), 0, static_cast<std::size_t>(memory.size));
}

template <typename T>
void writeMemory(const vk::raii::Device &device, BoundMemoryInfo memory, const std::vector<T> &values) {
    const auto byteSize = static_cast<vk::DeviceSize>(values.size() * sizeof(T));
    if (memory.memory == nullptr || byteSize > memory.size) {
        throw std::runtime_error("Mapped memory write exceeds allocation size");
    }
    const MappedMemory mapped(device, memory);
    std::copy(values.begin(), values.end(), static_cast<T *>(mapped.data()));
}

template <typename T>
std::vector<T> readMemory(const vk::raii::Device &device, BoundMemoryInfo memory, std::size_t elementCount) {
    const auto byteSize = static_cast<vk::DeviceSize>(elementCount * sizeof(T));
    if (memory.memory == nullptr || byteSize > memory.size) {
        throw std::runtime_error("Mapped memory read exceeds allocation size");
    }
    const MappedMemory mapped(device, memory);
    const auto *first = static_cast<const T *>(mapped.data());
    return {first, first + elementCount};
}

inline ResourceRequirements bufferRequirements(vk::DeviceSize byteSize) {
    ResourceRequirements requirements;
    requirements.kind = ResourceKind::StorageBuffer;
    requirements.descriptorType = vk::DescriptorType::eStorageBuffer;
    requirements.buffer.byteSize = byteSize;
    requirements.buffer.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    return requirements;
}

inline constexpr std::string_view addBuffersGlsl = R"(
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Lhs { int values[]; } lhs;
layout(set = 0, binding = 1) readonly buffer Rhs { int values[]; } rhs;
layout(set = 0, binding = 2) writeonly buffer Output { int values[]; } outputBuffer;

void main() {
    uint index = gl_GlobalInvocationID.x;
    outputBuffer.values[index] = lhs.values[index] + rhs.values[index];
}
)";

// [compute-description-begin]
inline ComputeShaderDescription addBuffersDescription(std::size_t elementCount) {
    ComputeShaderDescription description;
    description.module.codeKind = ModuleCodeKind::Glsl;
    description.module.source = addBuffersGlsl;
    description.dispatch = {static_cast<uint32_t>(elementCount), 1, 1};

    const auto byteSize = static_cast<vk::DeviceSize>(elementCount * sizeof(int32_t));
    description.resources = {
        {"lhs", 0, 0, ResourceAccess::Read, bufferRequirements(byteSize)},
        {"rhs", 0, 1, ResourceAccess::Read, bufferRequirements(byteSize)},
        {"output", 0, 2, ResourceAccess::Write, bufferRequirements(byteSize)},
    };
    return description;
}
// [compute-description-end]

// [vgf-building-begin]
inline std::string addBuffersVgf(std::size_t elementCount, bool embedImplementation = true) {
    if (elementCount > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("The element count exceeds the VGF dispatch limit");
    }

    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    const auto module = embedImplementation
                            ? encoder->AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main",
                                                 mlsdk::vgflib::ShaderType::GLSL, std::string(addBuffersGlsl))
                            : encoder->AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main");

    const auto shape = std::vector<int64_t>{static_cast<int64_t>(elementCount)};
    const auto strides = std::vector<int64_t>{static_cast<int64_t>(sizeof(int32_t))};
    const auto lhs = encoder->AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, shape, strides);
    const auto rhs = encoder->AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, shape, strides);
    const auto output =
        encoder->AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, shape, strides);

    const auto lhsBinding = encoder->AddBindingSlot(0, lhs);
    const auto rhsBinding = encoder->AddBindingSlot(1, rhs);
    const auto outputBinding = encoder->AddBindingSlot(2, output);
    const auto descriptorSet = encoder->AddDescriptorSetInfo({lhsBinding, rhsBinding, outputBinding}, 0);
    const std::vector<mlsdk::vgflib::GraphConstantBindingRef> noGraphConstants;
    encoder->AddSegmentInfo(module, "add_int32_buffers_segment", {descriptorSet}, {lhsBinding, rhsBinding},
                            {outputBinding}, noGraphConstants, {static_cast<uint32_t>(elementCount), 1, 1});
    encoder->AddModelSequenceInputsOutputs({lhsBinding, rhsBinding}, {"lhs", "rhs"}, {outputBinding}, {"output"});
    encoder->Finish();

    std::stringstream stream;
    if (!encoder->WriteTo(stream)) {
        throw std::runtime_error("Failed to encode the sample VGF");
    }
    return stream.str();
}
// [vgf-building-end]

} // namespace mlworkloadlib::samples
