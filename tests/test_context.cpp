/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib_utils/application_context.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_raii.hpp>

namespace {

using namespace mlsdk::workloadlib;

class ComputeOnlyContextTest : public ::testing::Test, protected utils::ApplicationContext {
  protected:
    ComputeOnlyContextTest() { requirements_.capabilities = RuntimeDeviceCapability::Compute; }

    void SetUp() override {
        if (!initialize(requirements_)) {
            GTEST_SKIP() << "No Vulkan device supports compute-only context requirements";
        }
    }

    RuntimeContextDeviceRequirements requirements_;
};

TEST_F(ComputeOnlyContextTest, CreatesRuntimeContext) {
    auto context = Context::create(requirements_);
    const auto view = context.contextView();
    const auto queueFamilies = view.physicalDevice.get().getQueueFamilyProperties();
    ASSERT_LT(view.queueFamilyIndex, queueFamilies.size());
    EXPECT_EQ(queueFamilies[view.queueFamilyIndex].queueFlags & vk::QueueFlagBits::eCompute,
              vk::QueueFlagBits::eCompute);
}

} // namespace
