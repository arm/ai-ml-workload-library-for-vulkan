/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib_utils/application_context.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <limits>
#include <utility>

namespace mlsdk::workloadlib::test {

template <typename Function> class ScopedDeviceDispatcherHook {
  public:
    ScopedDeviceDispatcherHook(vk::raii::Device &device, Function vk::raii::detail::DeviceDispatcher::*entry,
                               Function replacement)
        // Vulkan-Hpp exposes only a const dispatcher, even for a mutable device.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        : entry_(const_cast<vk::raii::detail::DeviceDispatcher &>(*device.getDispatcher()).*entry),
          original_(std::exchange(entry_, replacement)) {}

    ~ScopedDeviceDispatcherHook() { entry_ = original_; }

    ScopedDeviceDispatcherHook(const ScopedDeviceDispatcherHook &) = delete;
    ScopedDeviceDispatcherHook &operator=(const ScopedDeviceDispatcherHook &) = delete;
    ScopedDeviceDispatcherHook(ScopedDeviceDispatcherHook &&) = delete;
    ScopedDeviceDispatcherHook &operator=(ScopedDeviceDispatcherHook &&) = delete;

    Function original() const { return original_; }

  private:
    Function &entry_;
    Function original_;
};

template <typename RecordCommands>
void recordAndSubmitCommands(const vk::raii::Device &device, const vk::raii::Queue &queue, vk::CommandPool commandPool,
                             RecordCommands recordCommands) {
    auto commandBuffer =
        std::move(device.allocateCommandBuffers({commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    recordCommands(*commandBuffer);
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);
}

template <typename RecordCommands>
void recordAndSubmitCommands(const vk::raii::Device &device, const vk::raii::Queue &queue, uint32_t queueFamilyIndex,
                             RecordCommands recordCommands) {
    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    recordAndSubmitCommands(device, queue, *commandPool, std::move(recordCommands));
}

/*******************************************************************************
 * Vulkan fixture
 *******************************************************************************/

class RuntimeSessionExecutionTest : public ::testing::Test, protected utils::ApplicationContext {
  protected:
    RuntimeSessionExecutionTest() = default;

    void SetUp() override {
        if (!initialize()) {
            GTEST_SKIP() << "No Vulkan device with required data graph extensions and compute queue support";
        }
    }

    Context wrappedContext() { return Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue}); }
};

} // namespace mlsdk::workloadlib::test
