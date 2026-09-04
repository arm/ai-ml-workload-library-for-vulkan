/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vulkan/vulkan.hpp>

#include <optional>

namespace mlworkloadlib {

/*******************************************************************************
 * Binding metadata
 *******************************************************************************/

// Memory backing information for an externally supplied Vulkan resource.
// Required only when aliasing or runtime-created peer resources need compatible
// memory.
struct BoundMemoryInfo {
    vk::DeviceMemory memory = nullptr;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
};

// Tensor handle and optional memory information for tensor binding.
struct TensorBindingInfo {
    vk::TensorARM tensor = nullptr;
    BoundMemoryInfo memory{};
};

// Buffer handle and optional memory information for storage-buffer binding.
struct BufferBindingInfo {
    vk::Buffer buffer = nullptr;
    BoundMemoryInfo memory{};
};

// Image descriptor and optional memory information for image binding. The
// subresource range must match the image requirements for the bound resource.
// If layout is set, PreparedExecution treats it as the current image layout and
// records the transition to the workload-required descriptor layout before first
// use. Otherwise image layout is caller-managed.
struct ImageBindingInfo {
    vk::Image image = nullptr;
    BoundMemoryInfo memory{};

    vk::ImageView imageView = nullptr;
    vk::Sampler sampler = nullptr;
    std::optional<vk::ImageLayout> layout;
    vk::ImageSubresourceRange subresourceRange;
};

} // namespace mlworkloadlib
