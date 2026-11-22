/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/runtime_resource.hpp"
#include "internal/workload_builder.hpp"
#include "test_resource_requirements.hpp"
#include "test_spirv_utils.hpp"
#include "test_vulkan_fixture.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib_utils/mapped_device_memory.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_beta.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlsdk::workloadlib;

TEST(ScaledTensorRuntimeMetadata, ValidatesExplicitStrides) {
    const auto format = vk::Format::eR8SfloatFpencodingFloat8E4M3ARM;
    const std::vector<int64_t> shape{2, 64};
    const std::vector<int64_t> scaleShape{2, 2};
    const ScaledTensorConfig config{{32}, TensorScaleFormat::E8M0};
    const auto validate = [&](const std::vector<int64_t> &strides) {
        return detail::validateScaledTensorRuntimeMetadata(format, shape, strides, scaleShape, config);
    };

    EXPECT_EQ(validate({128, 1}), (std::vector<int64_t>{4, 1}));
    EXPECT_THROW((void)validate({128, 2}), std::runtime_error);
    EXPECT_THROW((void)validate({32, 1}), std::runtime_error);
    EXPECT_THROW((void)validate({65, 1}), std::runtime_error);

#ifdef VK_ARM_tensors2
    EXPECT_EQ(detail::validateScaledTensorRuntimeMetadata(vk::Format::eOcpmxfp32SfloatBlockFpencodingFloat4E2M1ARM,
                                                          shape, {64, 16}, scaleShape, config),
              (std::vector<int64_t>{4, 1}));
#endif
}

#ifdef VK_ARM_tensors2
using namespace mlsdk::workloadlib::test;

Workload scaledTensorWorkload() {
    auto requirements = tensorRequirements(vk::Format::eR8SfloatFpencodingFloat8E4M3ARM, {64});
    requirements.tensor.usage = vk::TensorUsageFlagBitsARM::eShader;
    requirements.tensor.scaledTensorConfig = ScaledTensorConfig{{32}, TensorScaleFormat::E8M0};
    detail::WorkloadBuilder builder;
    builder.addResource("input", requirements, detail::Resource::Role::Input);
    builder.addResource("output", requirements, detail::Resource::Role::Output);
    return builder.finish();
}

class ScaledRuntimeAllocationTest : public RuntimeSessionExecutionTest {
  protected:
    ScaledRuntimeAllocationTest() {
        features_.tensorBlockScaled = VK_TRUE;
        requirements_.capabilities = RuntimeDeviceCapability::Compute | RuntimeDeviceCapability::Tensors;
        requirements_.requiredDeviceExtensions.push_back(VK_ARM_TENSORS_2_EXTENSION_NAME);
        requirements_.deviceFeaturePNext = &features_;
    }

    void SetUp() override {
        if (!initialize(requirements_)) {
            GTEST_SKIP() << "No Vulkan device supports the required scaled tensor capabilities";
        }
    }

    vk::PhysicalDeviceTensors2FeaturesARM features_;
    RuntimeContextDeviceRequirements requirements_;
};

class TensorPlaneCalls {
  public:
    explicit TensorPlaneCalls(const vk::raii::Device &device)
        : get_(device, &vk::raii::detail::DeviceDispatcher::vkGetTensorMemoryRequirementsARM, get),
          bind_(device, &vk::raii::detail::DeviceDispatcher::vkBindTensorMemoryARM, bind),
          free_(device, &vk::raii::detail::DeviceDispatcher::vkFreeMemory, free),
          destroy_(device, &vk::raii::detail::DeviceDispatcher::vkDestroyTensorARM, destroy) {
        active_ = this;
    }

    ~TensorPlaneCalls() { active_ = nullptr; }
    TensorPlaneCalls(const TensorPlaneCalls &) = delete;
    TensorPlaneCalls &operator=(const TensorPlaneCalls &) = delete;

    bool failSecondBind = false;
    std::vector<uint32_t> queries;
    std::vector<uint32_t> binds;
    std::vector<char> releases;

  private:
    static uint32_t planeIndex(const void *next) {
        const auto *plane = static_cast<const VkTensorPlaneInfoARM *>(next);
        if (plane == nullptr || plane->sType != VK_STRUCTURE_TYPE_TENSOR_PLANE_INFO_ARM) {
            ADD_FAILURE() << "Scaled tensor memory call is missing its plane selector";
            return std::numeric_limits<uint32_t>::max();
        }
        return plane->planeIndex;
    }

    static void VKAPI_CALL get(VkDevice device, const VkTensorMemoryRequirementsInfoARM *info,
                               VkMemoryRequirements2 *result) {
        active_->queries.push_back(planeIndex(info->pNext));
        active_->get_.original()(device, info, result);
    }

    static VkResult VKAPI_CALL bind(VkDevice device, uint32_t count, const VkBindTensorMemoryInfoARM *infos) {
        for (uint32_t i = 0; i < count; ++i) {
            active_->binds.push_back(planeIndex(infos[i].pNext));
        }
        if (active_->failSecondBind && active_->binds.size() == 2) {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return active_->bind_.original()(device, count, infos);
    }

    static void VKAPI_CALL free(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *callbacks) {
        active_->releases.push_back('M');
        active_->free_.original()(device, memory, callbacks);
    }

    static void VKAPI_CALL destroy(VkDevice device, VkTensorARM tensor, const VkAllocationCallbacks *callbacks) {
        active_->releases.push_back('T');
        active_->destroy_.original()(device, tensor, callbacks);
    }

    ScopedDispatcherHook<PFN_vkGetTensorMemoryRequirementsARM> get_;
    ScopedDispatcherHook<PFN_vkBindTensorMemoryARM> bind_;
    ScopedDispatcherHook<PFN_vkFreeMemory> free_;
    ScopedDispatcherHook<PFN_vkDestroyTensorARM> destroy_;
    inline static thread_local TensorPlaneCalls *active_ = nullptr;
};

TEST_F(ScaledRuntimeAllocationTest, AllocatesBothPlanesAndPreservesMoves) {
    auto workload = scaledTensorWorkload();
    auto context = Context::create(requirements_);
    auto input = context.createTensor(workload.resource(0));
    ASSERT_EQ(input.planeCount(), 2u);
    EXPECT_THROW((void)input.memory(), std::runtime_error);
    EXPECT_NE(input.planeMemory(0).memory, input.planeMemory(1).memory);
    EXPECT_GT(input.planeMemory(0).size, 0u);
    EXPECT_GT(input.planeMemory(1).size, 0u);

    const std::vector<uint8_t> values(64, 0x38);
    const std::vector<uint8_t> scales = {0x7F, 0x80};
    utils::writeDeviceMemory(context.contextView().device, input.planeMemory(0), values);
    utils::writeDeviceMemory(context.contextView().device, input.planeMemory(1), scales);

    const auto valueMemory = input.planeMemory(0).memory;
    const auto scaleMemory = input.planeMemory(1).memory;
    TensorAllocation moved(std::move(input));
    EXPECT_EQ(input.planeCount(), 0u); // NOLINT(bugprone-use-after-move): verify moved-from state.
    auto assigned = context.createTensor(workload.resource(1));
    EXPECT_EQ(assigned.planeCount(), 2u);
    assigned = std::move(moved);
    EXPECT_EQ(assigned.planeCount(), 2u);
    EXPECT_EQ(assigned.planeMemory(0).memory, valueMemory);
    EXPECT_EQ(assigned.planeMemory(1).memory, scaleMemory);
    EXPECT_EQ(utils::readDeviceMemory<uint8_t>(context.contextView().device, assigned.planeMemory(0), values.size()),
              values);
    EXPECT_EQ(utils::readDeviceMemory<uint8_t>(context.contextView().device, assigned.planeMemory(1), scales.size()),
              scales);
}

TEST_F(ScaledRuntimeAllocationTest, FailedSecondPlaneBindReleasesBothAllocations) {
    auto workload = scaledTensorWorkload();
    auto context = Context::create(requirements_);
    TensorPlaneCalls calls(context.contextView().device);
    calls.failSecondBind = true;

    EXPECT_THROW((void)context.createTensor(workload.resource(0)), vk::SystemError);
    EXPECT_EQ(calls.queries, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(calls.binds, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(calls.releases, (std::vector<char>{'T', 'M', 'M'}));
}

TEST_F(ScaledRuntimeAllocationTest, AllocatesBothPlanesForUnaliasedIntermediate) {
    detail::WorkloadBuilder builder;
    auto requirements = tensorRequirements(vk::Format::eR8SfloatFpencodingFloat8E4M3ARM, {64});
    requirements.tensor.usage = vk::TensorUsageFlagBitsARM::eShader;
    requirements.tensor.scaledTensorConfig = ScaledTensorConfig{{32}, TensorScaleFormat::E8M0};
    const auto intermediate = builder.addResource("intermediate", requirements, detail::Resource::Role::Intermediate);
    ModuleImplementation module;
    module.codeKind = ModuleCodeKind::Spirv;
    module.spirv = assembleSpirv(readSpvasmFile(ML_WORKLOAD_LIB_NO_DESCRIPTORS_SPVASM));
    const auto moduleIndex = builder.addModule(std::move(module), "noop", "main");
    const auto executable = builder.addExecutable("noop", ExecutableKind::Compute, moduleIndex);
    builder.setDispatchShape(executable, std::array<uint32_t, 3>{1, 1, 1});
    builder.addDescriptorBinding(executable, intermediate, 0, 0, ResourceAccess::ReadWrite);
    auto workload = builder.finish();

    auto context = Context::create(requirements_);
    Session session(context, workload);
    session.configure();
    TensorPlaneCalls calls(context.contextView().device);
    {
        auto bindings = session.createBindingSet();
        auto execution = session.prepare(bindings);
        EXPECT_EQ(calls.queries, (std::vector<uint32_t>{0, 1}));
        EXPECT_EQ(calls.binds, (std::vector<uint32_t>{0, 1}));
        EXPECT_TRUE(calls.releases.empty());
        (void)execution;
    }
    EXPECT_EQ(calls.releases, (std::vector<char>{'T', 'M', 'M'}));
}

#endif
} // namespace
