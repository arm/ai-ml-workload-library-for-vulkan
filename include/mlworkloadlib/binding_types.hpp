/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vulkan/vulkan.hpp>

#include <optional>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Binding metadata
 *******************************************************************************/

/**
 * @brief Non-owning memory information for a bound Vulkan resource.
 *
 * Required when ResourceRequirementsView::requiresBoundMemoryInfo() is true.
 */
struct BoundMemoryInfo {
    vk::DeviceMemory memory = nullptr;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
};

/** @brief Tensor handle and optional memory information for tensor binding. */
struct TensorBindingInfo {
    vk::TensorARM tensor = nullptr;
    BoundMemoryInfo memory{};
};

/** @brief Buffer handle and optional memory information for storage-buffer binding. */
struct BufferBindingInfo {
    vk::Buffer buffer = nullptr;
    BoundMemoryInfo memory{};
};

/**
 * @brief Image descriptor and optional memory information for image binding.
 *
 * The subresource range must match
 * ImageRequirementsView::requiredSubresourceRange(). If @ref layout has a
 * value, PreparedExecution transitions the image from that layout to
 * ImageRequirementsView::requiredLayout() before its first workload use. If it
 * has no value, image layout transitions remain the caller's responsibility.
 */
struct ImageBindingInfo {
    vk::Image image = nullptr;
    BoundMemoryInfo memory{};

    vk::ImageView imageView = nullptr;
    vk::Sampler sampler = nullptr;
    std::optional<vk::ImageLayout> layout;
    vk::ImageSubresourceRange subresourceRange;
};

} // namespace mlsdk::workloadlib
