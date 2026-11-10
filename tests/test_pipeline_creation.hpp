/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "test_vulkan_fixture.hpp"

#include "mlworkloadlib/session.hpp"

namespace mlsdk::workloadlib::test {

// Create real pipelines, then inject results to test error handling and cleanup.
// Record cache forwarding and pipeline creation/destruction calls.
class PipelineCreationHook {
  public:
    explicit PipelineCreationHook(vk::raii::Device &device)
        : compute_(device, &vk::raii::detail::DeviceDispatcher::vkCreateComputePipelines, createCompute),
          graph_(device, &vk::raii::detail::DeviceDispatcher::vkCreateDataGraphPipelinesARM, createGraph),
          destroy_(device, &vk::raii::detail::DeviceDispatcher::vkDestroyPipeline, destroy) {
        active_ = this;
    }
    ~PipelineCreationHook() { active_ = nullptr; }

    PipelineCreationHook(const PipelineCreationHook &) = delete;
    PipelineCreationHook &operator=(const PipelineCreationHook &) = delete;
    PipelineCreationHook(PipelineCreationHook &&) = delete;
    PipelineCreationHook &operator=(PipelineCreationHook &&) = delete;

    VkResult result = VK_SUCCESS;
    unsigned creations = 0;
    unsigned destructions = 0;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::vector<VkPipelineCache> caches;

  private:
    static VKAPI_ATTR VkResult VKAPI_CALL createCompute(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                        const VkComputePipelineCreateInfo *infos,
                                                        const VkAllocationCallbacks *allocator, VkPipeline *pipelines) {
        ++active_->creations;
        active_->cache = cache;
        active_->caches.push_back(cache);
        const auto result = active_->compute_.original()(device, cache, count, infos, allocator, pipelines);
        return result == VK_SUCCESS ? active_->result : result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL createGraph(VkDevice device, VkDeferredOperationKHR deferredOperation,
                                                      VkPipelineCache cache, uint32_t count,
                                                      const VkDataGraphPipelineCreateInfoARM *infos,
                                                      const VkAllocationCallbacks *allocator, VkPipeline *pipelines) {
        ++active_->creations;
        active_->cache = cache;
        active_->caches.push_back(cache);
        const auto result =
            active_->graph_.original()(device, deferredOperation, cache, count, infos, allocator, pipelines);
        return result == VK_SUCCESS ? active_->result : result;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice device, VkPipeline pipeline,
                                              const VkAllocationCallbacks *allocator) {
        ++active_->destructions;
        active_->destroy_.original()(device, pipeline, allocator);
    }
    inline static thread_local PipelineCreationHook *active_ = nullptr;
    ScopedDeviceDispatcherHook<PFN_vkCreateComputePipelines> compute_;
    ScopedDeviceDispatcherHook<PFN_vkCreateDataGraphPipelinesARM> graph_;
    ScopedDeviceDispatcherHook<PFN_vkDestroyPipeline> destroy_;
};

inline void verifyPipelineCreation(Context &context, const Workload &workload, vk::raii::Device &device) {
    const vk::raii::PipelineCache cache(device, vk::PipelineCacheCreateInfo{});
    PipelineCreationHook hook(device);
    SessionOptions options;
    options.pipelineCache = &cache;
    {
        SCOPED_TRACE("driver error with a pipeline to clean up");
        hook.result = VK_ERROR_OUT_OF_HOST_MEMORY;
        const auto destroyedBefore = hook.destructions;
        const auto createdBefore = hook.creations;
        Session session(context, workload, options);
        EXPECT_THROW(session.configure(), vk::OutOfHostMemoryError);
        EXPECT_EQ(hook.creations, createdBefore + 1);
        EXPECT_EQ(hook.destructions, destroyedBefore + 1);
    }

    {
        SCOPED_TRACE("failed compilation allows retry, successful compilation allows reuse");
        CompiledExecution compiled(context, workload, options);
        const auto createdBefore = hook.creations;
        const auto destroyedBefore = hook.destructions;
        hook.result = VK_PIPELINE_COMPILE_REQUIRED;
        {
            Session failed(context, workload, compiled);
            try {
                failed.configure();
                ADD_FAILURE() << "Compile-required result was accepted";
            } catch (const vk::SystemError &error) {
                EXPECT_EQ(error.code(), vk::make_error_code(vk::Result::ePipelineCompileRequired));
            }
        }
        EXPECT_EQ(hook.creations, createdBefore + 1);
        EXPECT_EQ(hook.destructions, destroyedBefore + 1);

        hook.result = VK_SUCCESS;
        {
            Session retry(context, workload, compiled);
            ASSERT_NO_THROW(retry.configure());
        }
        EXPECT_EQ(hook.creations, createdBefore + 2);
        EXPECT_EQ(hook.cache, static_cast<VkPipelineCache>(*cache));
        // Shared compiled state keeps the successful pipeline alive after its Session is destroyed.
        EXPECT_EQ(hook.destructions, destroyedBefore + 1);
        {
            Session reuse(context, workload, compiled);
            EXPECT_NO_THROW(reuse.configure());
        }
        EXPECT_EQ(hook.creations, createdBefore + 2);
    }
    const vk::raii::PipelineCache overrideCache(device, vk::PipelineCacheCreateInfo{});
    for (const auto *selected : {&overrideCache, static_cast<const vk::raii::PipelineCache *>(nullptr)}) {
        SCOPED_TRACE(selected ? "override cache" : "explicitly disabled cache");
        CompiledExecution compiled(context, workload, options);
        Session session(context, workload, compiled);
        const auto executable = workload.executable(0);
        session.setExecutableOptions(executable, {selected});
        ASSERT_NO_THROW(session.configure());
        EXPECT_EQ(hook.cache, selected ? static_cast<VkPipelineCache>(**selected) : VK_NULL_HANDLE);
        EXPECT_THROW(session.setExecutableOptions(executable, {}), std::runtime_error);
        Session reuse(context, workload, compiled);
        EXPECT_THROW(reuse.setExecutableOptions(executable, {}), std::runtime_error);
    }
    {
        CompiledExecution compiled(context, workload, options);
        Session pending(context, workload, compiled);
        pending.setExecutableOptions(workload.executable(0), {});
        Session first(context, workload, compiled);
        ASSERT_NO_THROW(first.configure());
        const auto createdBefore = hook.creations;
        EXPECT_THROW(pending.configure(), std::runtime_error);
        EXPECT_THROW(pending.setExecutableOptions(workload.executable(0), {}), std::runtime_error);
        EXPECT_EQ(hook.creations, createdBefore);
    }
    // The default session options must still forward a null application cache.
    Session session(context, workload);
    EXPECT_NO_THROW(session.configure());
    EXPECT_EQ(hook.cache, VK_NULL_HANDLE);
}

} // namespace mlsdk::workloadlib::test
