/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/workload.hpp"
#include "mlworkloadlib_utils/mapped_device_memory.hpp"
#include "mlworkloadlib_utils/workload_metadata.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mlsdk::workloadlib::samples {

inline constexpr std::string_view addBuffersGlsl = R"(
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Lhs { int values[]; } lhs;
layout(set = 0, binding = 1) readonly buffer Rhs { int values[]; } rhs;
layout(set = 0, binding = 2) writeonly buffer Output { int values[]; } outputBuffer;

void main() {
    uint index = gl_GlobalInvocationID.x;
    outputBuffer.values[index] = lhs.values[index] + rhs.values[index];
}
)";

// [compute-description-begin]
inline ComputeShaderDescription addBuffersDescription(std::size_t elementCount) {
    ComputeShaderDescription description;
    description.module.codeKind = ModuleCodeKind::Glsl;
    description.module.source = addBuffersGlsl;
    description.dispatch = {static_cast<uint32_t>(elementCount), 1, 1};

    const auto byteSize = static_cast<vk::DeviceSize>(elementCount * sizeof(int32_t));
    description.resources = {
        {"lhs", 0, 0, ResourceAccess::Read, utils::bufferRequirements(byteSize)},
        {"rhs", 0, 1, ResourceAccess::Read, utils::bufferRequirements(byteSize)},
        {"output", 0, 2, ResourceAccess::Write, utils::bufferRequirements(byteSize)},
    };
    return description;
}
// [compute-description-end]

} // namespace mlsdk::workloadlib::samples
