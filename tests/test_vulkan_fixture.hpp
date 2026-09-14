/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib_utils/application_context.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace mlsdk::workloadlib::test {

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
    RuntimeSessionExecutionTest() : utils::ApplicationContext(utils::ApplicationContext::DeferredInitialization{}) {}

    void SetUp() override {
        if (!initialize("mlworkloadlib-test")) {
            GTEST_SKIP() << "No Vulkan device with required data graph extensions and compute queue support";
        }
    }

    Context wrappedContext() { return Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue}); }
};

} // namespace mlsdk::workloadlib::test
