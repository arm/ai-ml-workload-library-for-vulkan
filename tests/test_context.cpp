/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "standalone_compute/test_standalone_compute_utils.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/workload.hpp"
#include "mlworkloadlib_utils/application_context.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_raii.hpp>

#include <type_traits>
#include <utility>

namespace {

using namespace mlsdk::workloadlib;

static_assert(!std::is_move_assignable_v<RuntimeAllocation>);
static_assert(std::is_move_assignable_v<TensorAllocation>);
static_assert(std::is_move_assignable_v<BufferAllocation>);
static_assert(std::is_move_assignable_v<ImageAllocation>);
static_assert(std::is_nothrow_move_constructible_v<TensorAllocation>);
static_assert(std::is_nothrow_move_constructible_v<BufferAllocation>);
static_assert(std::is_nothrow_move_constructible_v<ImageAllocation>);

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

TEST_F(ComputeOnlyContextTest, AllocatesBufferAndImage) {
    auto description = test::makeAddInt32BuffersDescription();
    description.resources.push_back({"image", 0, 3, ResourceAccess::Read, test::makeSampledImageRequirements(false)});
    auto workload = Workload::fromComputeShader(std::move(description));
    auto context = Context::create(requirements_);

    auto buffer = context.createBuffer(workload.resource(0));
    const auto bufferHandle = buffer.handle();
    const auto bufferMemory = buffer.memory().memory;
    EXPECT_NE(bufferHandle, nullptr);
    EXPECT_NE(bufferMemory, nullptr);

    BufferAllocation movedBuffer(std::move(buffer));
    EXPECT_EQ(movedBuffer.handle(), bufferHandle);
    EXPECT_EQ(movedBuffer.memory().memory, bufferMemory);
    EXPECT_EQ(buffer.handle(), nullptr); // NOLINT(bugprone-use-after-move): verify the moved-from state.
    EXPECT_EQ(buffer.memory().memory, nullptr);

    auto image = context.createImage(workload.resource(3));
    const auto imageHandle = image.handle();
    const auto imageMemory = image.memory().memory;
    const auto imageView = image.binding().imageView;
    EXPECT_NE(imageHandle, nullptr);
    EXPECT_NE(imageMemory, nullptr);
    EXPECT_NE(imageView, nullptr);

    ImageAllocation movedImage(std::move(image));
    EXPECT_EQ(movedImage.handle(), imageHandle);
    EXPECT_EQ(movedImage.memory().memory, imageMemory);
    EXPECT_EQ(movedImage.binding().imageView, imageView);
    EXPECT_EQ(image.handle(), nullptr); // NOLINT(bugprone-use-after-move): verify the moved-from state.
    EXPECT_EQ(image.memory().memory, nullptr);
    EXPECT_EQ(image.binding().image, nullptr);
    EXPECT_EQ(image.binding().imageView, nullptr);
}

} // namespace
