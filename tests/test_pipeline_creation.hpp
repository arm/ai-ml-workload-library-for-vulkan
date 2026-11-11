/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "test_vulkan_fixture.hpp"

#include "mlworkloadlib/session.hpp"

namespace mlsdk::workloadlib::test {

// Create real pipelines, then inject results to test error handling and cleanup.
// Optionally skip the driver call to inspect inputs that the driver cannot load.
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
    unsigned destructions = 0;
    unsigned failAt = 0;
    bool validFeedback = true;
    bool skipDriverCall = false;

    struct CreationCall {
        VkPipelineCache cache;
        bool failOnCompileRequired;
        bool hasBinary = false;
        bool hasFeedback = false;
        uint32_t stageFeedbackCount = 0;
    };
    std::vector<CreationCall> calls;

  private:
    void recordAndPopulateFeedback(const void *chain) {
        auto &call = calls.back();
        const auto *node = static_cast<const VkBaseInStructure *>(chain);
        const VkPipelineCreationFeedbackCreateInfo *feedback = nullptr;
        while (node) {
#ifdef VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME
            if (node->sType == VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_BINARY_CREATE_INFO_ARM) {
                call.hasBinary = true;
            }
#endif
            if (node->sType == VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO) {
                feedback = reinterpret_cast<const VkPipelineCreationFeedbackCreateInfo *>(node);
            }
            node = node->pNext;
        }
        call.hasFeedback = feedback != nullptr;
        if (!feedback) {
            return;
        }
        ASSERT_NE(feedback->pPipelineCreationFeedback, nullptr);
        call.stageFeedbackCount = feedback->pipelineStageCreationFeedbackCount;
        if (call.stageFeedbackCount != 0) {
            ASSERT_NE(feedback->pPipelineStageCreationFeedbacks, nullptr);
        }
        *feedback->pPipelineCreationFeedback = {
            validFeedback ? static_cast<VkPipelineCreationFeedbackFlags>(VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) : 0u,
            123};
        for (uint32_t i = 0; i < call.stageFeedbackCount; ++i) {
            feedback->pPipelineStageCreationFeedbacks[i] = {0, 456};
        }
    }

    template <typename CreateInfo, typename PolicyFlag, typename DriverCall>
    VkResult createPipeline(VkPipelineCache cache, uint32_t count, const CreateInfo *infos, PolicyFlag policyFlag,
                            VkPipeline *pipelines, DriverCall driverCall) {
        const auto policyMask = static_cast<decltype(infos->flags)>(policyFlag);
        calls.push_back({cache, bool(infos->flags & policyMask)});
        if (skipDriverCall) {
            recordAndPopulateFeedback(infos->pNext);
            for (uint32_t i = 0; i < count; ++i) {
                pipelines[i] = VK_NULL_HANDLE;
            }
            return result == VK_SUCCESS ? VK_ERROR_UNKNOWN : result;
        }
        // Inspect policy independently of driver support and create a real owned handle.
        auto info = *infos;
        info.flags &= ~policyMask;
        const auto driverResult = driverCall(&info);
        recordAndPopulateFeedback(infos->pNext);
        return driverResult == VK_SUCCESS && (!failAt || failAt == calls.size()) ? result : driverResult;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createCompute(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                        const VkComputePipelineCreateInfo *infos,
                                                        const VkAllocationCallbacks *allocator, VkPipeline *pipelines) {
        return active_->createPipeline(cache, count, infos, VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
                                       pipelines, [&](const auto *info) {
                                           return active_->compute_.original()(device, cache, count, info, allocator,
                                                                               pipelines);
                                       });
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createGraph(VkDevice device, VkDeferredOperationKHR deferredOperation,
                                                      VkPipelineCache cache, uint32_t count,
                                                      const VkDataGraphPipelineCreateInfoARM *infos,
                                                      const VkAllocationCallbacks *allocator, VkPipeline *pipelines) {
        return active_->createPipeline(cache, count, infos, VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
                                       pipelines, [&](const auto *info) {
                                           return active_->graph_.original()(device, deferredOperation, cache, count,
                                                                             info, allocator, pipelines);
                                       });
    }

    static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice device, VkPipeline pipeline,
                                              const VkAllocationCallbacks *allocator) {
        ++active_->destructions;
        active_->destroy_.original()(device, pipeline, allocator);
    }

    inline static thread_local PipelineCreationHook *active_ = nullptr;
    ScopedDispatcherHook<PFN_vkCreateComputePipelines> compute_;
    ScopedDispatcherHook<PFN_vkCreateDataGraphPipelinesARM> graph_;
    ScopedDispatcherHook<PFN_vkDestroyPipeline> destroy_;
};

} // namespace mlsdk::workloadlib::test
