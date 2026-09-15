/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <limits>
#include <string_view>

namespace mlsdk::workloadlib::utils {

// Caller-owned Vulkan objects suitable for Context::wrap().
class ApplicationContext {
  public:
    explicit ApplicationContext(std::string_view applicationName = "mlworkloadlib-sample");
    ~ApplicationContext() = default;

    ApplicationContext(const ApplicationContext &) = delete;
    ApplicationContext &operator=(const ApplicationContext &) = delete;
    ApplicationContext(ApplicationContext &&) = delete;
    ApplicationContext &operator=(ApplicationContext &&) = delete;

    const vk::raii::Instance &vulkanInstance() const noexcept;
    const vk::raii::PhysicalDevice &vulkanPhysicalDevice() const noexcept;
    const vk::raii::Device &vulkanDevice() const noexcept;
    const vk::raii::Queue &vulkanQueue() const noexcept;
    uint32_t vulkanQueueFamilyIndex() const noexcept;

  protected:
    struct DeferredInitialization {};

    explicit ApplicationContext(DeferredInitialization) {}
    bool initialize(std::string_view applicationName);

    vk::raii::Context raiiContext;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue queue{nullptr};
    uint32_t queueFamilyIndex = std::numeric_limits<uint32_t>::max();
};

} // namespace mlsdk::workloadlib::utils
