/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "workload_impl.hpp"

#include "mlworkloadlib/compiled_execution.hpp"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace mlsdk::workloadlib::detail {

/*******************************************************************************
 * Compiled execution data
 *******************************************************************************/

struct CompiledExecutable {
    vk::raii::ShaderModule shaderModule{nullptr};
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout{nullptr};
    vk::raii::Pipeline pipeline{nullptr};
};

struct CompiledExecutionData {
    std::map<uint32_t, Module> moduleImplementations;
    std::vector<CompiledExecutable> compiledExecutables;
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

    Impl(Context &contextIn, const Workload &workloadIn, SessionOptions optionsIn)
        : context(&contextIn), workload(&workloadIn), options(optionsIn) {}

    /***************************************************************************
     * Stored state
     **************************************************************************/

    Context *context = nullptr;
    const Workload *workload = nullptr;
    SessionOptions options;
    std::optional<detail::CompiledExecutionData> compiledData;
};

} // namespace mlsdk::workloadlib
