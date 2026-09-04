/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "internal/utils.hpp"

#include "mlworkloadlib/workload.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace mlworkloadlib::test {

/*******************************************************************************
 * Workload resource requirements
 *******************************************************************************/

inline ResourceRequirements makeBufferRequirements(vk::DeviceSize byteSize) {
    ResourceRequirements resource;
    resource.kind = ResourceKind::StorageBuffer;
    resource.descriptorType = vk::DescriptorType::eStorageBuffer;
    resource.format = vk::Format::eR32Sint;
    resource.elementCount = byteSize / sizeof(int32_t);
    resource.buffer.byteSize = byteSize;
    resource.buffer.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    return resource;
}

inline ResourceRequirements makeTensorRequirements(vk::Format format, std::vector<int64_t> shape) {
    ResourceRequirements resource;
    resource.kind = ResourceKind::Tensor;
    resource.descriptorType = vk::DescriptorType::eTensorARM;
    resource.format = format;
    resource.elementCount = detail::utils::elementCount(shape);
    resource.tensor.shape = std::move(shape);
    resource.tensor.usage = vk::TensorUsageFlagBitsARM::eDataGraph;
    return resource;
}

inline ResourceRequirements makeStorageImageRequirements() {
    ResourceRequirements resource;
    resource.kind = ResourceKind::Image;
    resource.descriptorType = vk::DescriptorType::eStorageImage;
    resource.format = vk::Format::eR8G8B8A8Snorm;
    resource.elementCount = 16;
    resource.image.extent = vk::Extent3D(2, 2, 1);
    resource.image.usage = vk::ImageUsageFlagBits::eStorage;
    resource.image.requiredLayout = vk::ImageLayout::eGeneral;
    resource.image.range = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    return resource;
}

} // namespace mlworkloadlib::test
