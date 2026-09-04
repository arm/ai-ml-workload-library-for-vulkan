/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context_impl.hpp"
#include "workload_impl.hpp"

#include "mlworkloadlib/session.hpp"

#include <map>
#include <vector>

namespace mlworkloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct Session::Impl {
    using Module = detail::Module;

    /***************************************************************************
     * Executable state
     **************************************************************************/

    struct ExecutableState {
        uint32_t executableIndex = 0;
        vk::raii::ShaderModule shaderModule{nullptr};
        std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
        vk::raii::PipelineLayout pipelineLayout{nullptr};
        vk::raii::Pipeline pipeline{nullptr};
        // Members are destroyed in reverse declaration order. Keep sessionMemory before
        // graphSession so a graph session is destroyed before its bound memory.
        std::vector<vk::raii::DeviceMemory> sessionMemory;
        vk::raii::DataGraphPipelineSessionARM graphSession{nullptr};
    };

    /***************************************************************************
     * Lifetime
     **************************************************************************/

    Impl(Context &contextIn, const Workload &workloadIn)
        : workload(workloadIn), contextView(contextIn.contextImpl().contextView()) {}

    /***************************************************************************
     * Configuration
     **************************************************************************/

    void configure();

    /***************************************************************************
     * State access
     **************************************************************************/

    const Workload::Impl &workloadImpl() const noexcept { return workload.workloadImpl(); }

    /***************************************************************************
     * Stored state
     **************************************************************************/

    const Workload &workload;
    ContextView contextView;

    std::map<uint32_t, Module> moduleImplementations;

    std::vector<ExecutableState> executableStates;

    vk::raii::CommandPool commandPool{nullptr};
    vk::raii::CommandBuffer commandBuffer{nullptr};
    vk::raii::Fence fence{nullptr};

    bool configured = false;
};

} // namespace mlworkloadlib
