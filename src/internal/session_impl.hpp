/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compiled_execution_impl.hpp"
#include "context_impl.hpp"
#include "workload_impl.hpp"

#include "mlworkloadlib/session.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct Session::Impl {
    using Module = detail::Module;

    /***************************************************************************
     * Executable state
     **************************************************************************/

    struct ExecutableState {
        explicit ExecutableState(const detail::CompiledExecutable &compiledExecutableIn)
            : compiledExecutable(compiledExecutableIn) {}

        std::reference_wrapper<const detail::CompiledExecutable> compiledExecutable;

        // Members are destroyed in reverse declaration order. Keep sessionMemory before
        // graphSession so a graph session is destroyed before its bound memory.
        std::vector<vk::raii::DeviceMemory> sessionMemory;
        vk::raii::DataGraphPipelineSessionARM graphSession{nullptr};
        std::vector<DataGraphSessionMemoryInfo> sessionMemoryInfo;
    };

    /***************************************************************************
     * Lifetime
     **************************************************************************/

    Impl(Context &contextIn, const Workload &workloadIn, SessionOptions optionsIn);
    Impl(Context &contextIn, const Workload &workloadIn, CompiledExecution &compiledExecutionIn);

  private:
    /***************************************************************************
     * Configuration
     **************************************************************************/

    void createPipeline(detail::CompiledExecutable &compiledExecutable, uint32_t executableIndex) const;
    void configureExecutableState(uint32_t executableIndex);
    const ExecutableState &graphExecutableState(uint32_t executableIndex) const;
    void compileOrReuseExecutables();

  public:
    void configure();
    std::vector<vk::DataGraphPipelinePropertyARM> dataGraphPipelineProperties(uint32_t executableIndex) const;
    DataGraphPipelinePropertyData dataGraphPipelineProperty(uint32_t executableIndex,
                                                            vk::DataGraphPipelinePropertyARM property) const;
    uint32_t dataGraphSessionMemoryCount(uint32_t executableIndex) const;
    DataGraphSessionMemoryInfo dataGraphSessionMemory(uint32_t executableIndex, uint32_t memoryIndex) const;

    /***************************************************************************
     * Stored state
     **************************************************************************/

    const Workload &workload;
    ContextView contextView;

    // Keep compiled pipelines alive until after executableStates are destroyed.
    std::shared_ptr<CompiledExecution::Impl> compiledExecutionState;

    std::map<uint32_t, Module> moduleImplementations;

    std::vector<ExecutableState> executableStates;

    vk::raii::CommandPool commandPool{nullptr};
    vk::raii::CommandBuffer commandBuffer{nullptr};
    vk::raii::Fence fence{nullptr};

    bool configured = false;
};

} // namespace mlsdk::workloadlib
