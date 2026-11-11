/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_pipeline_creation.hpp"
#include "standalone_compute/test_standalone_compute_utils.hpp"
#include "standalone_data_graph/test_standalone_data_graph_utils.hpp"
#include "test_vulkan_fixture.hpp"

#include <gtest/gtest.h>

#include <optional>

namespace mlsdk::workloadlib::test {
namespace {

class PipelineCreationTest : public RuntimeSessionExecutionTest, public ::testing::WithParamInterface<ExecutableKind> {
  protected:
    Workload makeWorkload() const {
        if (GetParam() == ExecutableKind::Compute) {
            return Workload::fromComputeShader(makeAddInt32BuffersDescription());
        }
        return Workload::fromDataGraph(makeMaxpoolDescription());
    }
};

TEST_P(PipelineCreationTest, PipelineCreationFailureCleansUpReturnedHandle) {
    auto context = wrappedContext();
    auto workload = makeWorkload();
    PipelineCreationHook hook(device);
    hook.result = VK_ERROR_OUT_OF_HOST_MEMORY;
    Session session(context, workload);
    EXPECT_THROW(session.configure(), vk::OutOfHostMemoryError);
    EXPECT_EQ(hook.calls.size(), 1u);
    EXPECT_EQ(hook.destructions, 1u);
}

TEST_P(PipelineCreationTest, CompileRequiredAllowsRetry) {
    auto context = wrappedContext();
    auto workload = makeWorkload();
    for (const bool policy : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "failOnCompileRequired=" << policy);
        PipelineCreationHook hook(device);
        hook.result = VK_PIPELINE_COMPILE_REQUIRED;
        SessionOptions options;
        options.failOnCompileRequired = policy;
        CompiledExecution compiled(context, workload, options);
        {
            Session failed(context, workload, compiled);
            try {
                failed.configure();
                ADD_FAILURE() << "Compile-required result was accepted";
            } catch (const vk::SystemError &error) {
                EXPECT_EQ(error.code(), vk::make_error_code(vk::Result::ePipelineCompileRequired));
            }
        }
        ASSERT_EQ(hook.calls.size(), 1u);
        EXPECT_EQ(hook.destructions, 1u);
        EXPECT_EQ(hook.calls[0].failOnCompileRequired, policy);
        {
            // The Vulkan callback reads this result through the registered hook during retry.configure().
            // cppcheck-suppress unreadVariable
            hook.result = VK_SUCCESS;
            Session retry(context, workload, compiled);
            ASSERT_NO_THROW(retry.configure());
        }
        ASSERT_EQ(hook.calls.size(), 2u);
        EXPECT_EQ(hook.calls[1].failOnCompileRequired, policy);
        // Successful pipelines belong to the shared compiled state, not the Session.
        EXPECT_EQ(hook.destructions, 1u);
    }
}

TEST_P(PipelineCreationTest, PipelineCacheSelection) {
    auto context = wrappedContext();
    auto workload = makeWorkload();
    const vk::raii::PipelineCache cache(device, vk::PipelineCacheCreateInfo{});
    const vk::raii::PipelineCache overrideCache(device, vk::PipelineCacheCreateInfo{});
    struct Case {
        const char *name;
        const vk::raii::PipelineCache *sessionCache;
        std::optional<ExecutableOptions> executableOptions;
        const vk::raii::PipelineCache *expectedCache;
    };
    for (const auto &testCase :
         {Case{"default", nullptr, std::nullopt, nullptr}, Case{"session cache", &cache, std::nullopt, &cache},
          Case{"executable override", &cache, ExecutableOptions{&overrideCache}, &overrideCache},
          Case{"explicit null override", &cache, ExecutableOptions{}, nullptr}}) {
        SCOPED_TRACE(testCase.name);
        // Vulkan callbacks mutate this hook through its registered pointer.
        // NOLINTNEXTLINE(misc-const-correctness)
        PipelineCreationHook hook(device);
        SessionOptions options;
        options.pipelineCache = testCase.sessionCache;
        Session session(context, workload, options);
        if (testCase.executableOptions) {
            session.setExecutableOptions(workload.executable(0), *testCase.executableOptions);
        }
        ASSERT_NO_THROW(session.configure());
        ASSERT_EQ(hook.calls.size(), 1u);
        EXPECT_EQ(hook.calls[0].cache,
                  testCase.expectedCache ? static_cast<VkPipelineCache>(**testCase.expectedCache) : VK_NULL_HANDLE);
    }
}

TEST_P(PipelineCreationTest, ExecutableOptionsFreezeAfterCompilation) {
    auto context = wrappedContext();
    auto workload = makeWorkload();
    // Vulkan callbacks mutate this hook through its registered pointer.
    // NOLINTNEXTLINE(misc-const-correctness)
    PipelineCreationHook hook(device);
    CompiledExecution compiled(context, workload);
    const auto executable = workload.executable(0);
    Session pending(context, workload, compiled);
    pending.setExecutableOptions(executable, {});
    Session creator(context, workload, compiled);
    creator.setExecutableOptions(executable, {});
    ASSERT_NO_THROW(creator.configure());
    EXPECT_THROW(creator.setExecutableOptions(executable, {}), std::runtime_error);
    Session reuse(context, workload, compiled);
    EXPECT_THROW(reuse.setExecutableOptions(executable, {}), std::runtime_error);
    EXPECT_THROW(pending.configure(), std::runtime_error);
    EXPECT_THROW(pending.setExecutableOptions(executable, {}), std::runtime_error);
    EXPECT_EQ(hook.calls.size(), 1u);
}

TEST_P(PipelineCreationTest, PipelineFeedbackCollectionFollowsOptions) {
    auto context = wrappedContext();
    auto workload = makeWorkload();
    struct Case {
        bool collect;
        bool valid;
    };
    for (const auto testCase : {Case{false, true}, Case{true, true}, Case{true, false}}) {
        SCOPED_TRACE(::testing::Message() << "collect=" << testCase.collect << ", valid=" << testCase.valid);
        PipelineCreationHook hook(device);
        hook.validFeedback = testCase.valid;
        SessionOptions options;
        options.collectPipelineCreationFeedback = testCase.collect;
        Session session(context, workload, options);
        const auto executable = workload.executable(0);
        EXPECT_FALSE(session.pipelineCreationReport(executable));
        ASSERT_NO_THROW(session.configure());
        ASSERT_EQ(hook.calls.size(), 1u);
        const auto &call = hook.calls[0];
        EXPECT_FALSE(call.failOnCompileRequired);
        EXPECT_EQ(call.hasFeedback, testCase.collect);
        if (testCase.collect) {
            EXPECT_EQ(call.stageFeedbackCount, executable.type() == ExecutableKind::Compute ? 1u : 0u);
        }
        const auto report = session.pipelineCreationReport(executable);
        if (!testCase.collect) {
            EXPECT_FALSE(report);
            continue;
        }
        ASSERT_TRUE(report);
        EXPECT_EQ(report->pipelineFeedback.duration, 123u);
        EXPECT_EQ(report->pipelineFeedback.flags,
                  testCase.valid ? vk::PipelineCreationFeedbackFlagBits::eValid : vk::PipelineCreationFeedbackFlags{});
        ASSERT_EQ(report->stageFeedbacks.size(), executable.type() == ExecutableKind::Compute ? 1u : 0u);
        for (const auto &stage : report->stageFeedbacks) {
            EXPECT_FALSE(stage.flags);
            EXPECT_EQ(stage.duration, 456u);
        }
    }
}

TEST_P(PipelineCreationTest, PipelineReportsSurviveCreatorDestructionAndReuse) {
    auto context = wrappedContext();
    auto workload = makeWorkload();
    // Vulkan callbacks mutate this hook through its registered pointer.
    // NOLINTNEXTLINE(misc-const-correctness)
    PipelineCreationHook hook(device);
    SessionOptions options;
    options.collectPipelineCreationFeedback = true;
    CompiledExecution compiled(context, workload, options);
    const auto executable = workload.executable(0);
    {
        Session creator(context, workload, compiled);
        ASSERT_NO_THROW(creator.configure());
    }
    EXPECT_EQ(hook.destructions, 0u);
    Session reuse(context, workload, compiled);
    ASSERT_TRUE(reuse.pipelineCreationReport(executable));
    EXPECT_EQ(reuse.pipelineCreationReport(executable)->pipelineFeedback.duration, 123u);
    ASSERT_NO_THROW(reuse.configure());
    ASSERT_TRUE(reuse.pipelineCreationReport(executable));
    EXPECT_EQ(reuse.pipelineCreationReport(executable)->pipelineFeedback.duration, 123u);
    EXPECT_EQ(hook.calls.size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(WorkloadKinds, PipelineCreationTest,
                         ::testing::Values(ExecutableKind::Compute, ExecutableKind::Graph),
                         [](const ::testing::TestParamInfo<ExecutableKind> &info) {
                             return info.param == ExecutableKind::Compute ? "Compute" : "DataGraph";
                         });

} // namespace
} // namespace mlsdk::workloadlib::test
