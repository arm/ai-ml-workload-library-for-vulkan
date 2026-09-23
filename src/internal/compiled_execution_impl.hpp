/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "session_tiling.hpp"

#include "mlworkloadlib/compiled_execution.hpp"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <optional>
#include <vector>

namespace mlsdk::workloadlib::detail {

/*******************************************************************************
 * Compiled executable state
 *******************************************************************************/

struct CompiledExecutable {
    vk::raii::ShaderModule shaderModule{nullptr};
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout{nullptr};
    vk::raii::Pipeline pipeline{nullptr};
};

struct CompiledState {
    // Only graph tensors and their alias peers have a tiling entry.
    std::vector<std::optional<ResourceTiling>> pipelineTilings;
    std::vector<CompiledExecutable> executables;
};

} // namespace mlsdk::workloadlib::detail

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct CompiledExecution::Impl {
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    Impl(Context &contextIn, const Workload &workloadIn, SessionOptions optionsIn);

    /***************************************************************************
     * Stored state
     **************************************************************************/

    Context *context = nullptr;
    const Workload *workload = nullptr;
    SessionOptions options;

    std::optional<detail::CompiledState> compiledState;
};

} // namespace mlsdk::workloadlib
