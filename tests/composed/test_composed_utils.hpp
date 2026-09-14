/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_resource_requirements.hpp"
#include "test_spirv_utils.hpp"

#include "mlworkloadlib/workload.hpp"

#ifdef ML_WORKLOAD_LIB_ENABLE_VGF_SUPPORT
#    include "vgf/test_vgf_utils.hpp"
#endif

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <vector>

namespace mlsdk::workloadlib::test {

inline DispatchShape dispatchForNhwcTensor(const std::vector<int64_t> &shape) {
    return {static_cast<uint32_t>(shape.at(1)), static_cast<uint32_t>(shape.at(2)), static_cast<uint32_t>(shape.at(3))};
}

inline ComputeShaderDescription makeTensorAddDescription(std::vector<int64_t> shape) {
    ComputeShaderDescription description;
    description.module.codeKind = ModuleCodeKind::Glsl;
    description.module.source = R"(
#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly uniform tensorARM<int8_t, 4> firstInput;
layout(set = 0, binding = 2) readonly uniform tensorARM<int8_t, 4> secondInput;
layout(set = 0, binding = 3) writeonly uniform tensorARM<int8_t, 4> outputData;

void main() {
    uint coords[4] = uint[](0, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
    int8_t firstValue;
    int8_t secondValue;
    tensorReadARM(firstInput, coords, firstValue);
    tensorReadARM(secondInput, coords, secondValue);
    tensorWriteARM(outputData, coords, int8_t(firstValue + secondValue));
}
)";
    description.entryPoint = "main";
    description.dispatch = dispatchForNhwcTensor(shape);
    description.resources = {
        {"first_input", 0, 0, ResourceAccess::Read, tensorRequirements(vk::Format::eR8Sint, shape)},
        {"second_input", 0, 2, ResourceAccess::Read, tensorRequirements(vk::Format::eR8Sint, shape)},
        {"output", 0, 3, ResourceAccess::Write, tensorRequirements(vk::Format::eR8Sint, std::move(shape))},
    };
    return description;
}

inline ComputeShaderDescription makeTensorSubtractDescription(std::vector<int64_t> shape) {
    ComputeShaderDescription description;
    description.module.codeKind = ModuleCodeKind::Glsl;
    description.module.source = R"(
#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 1) readonly uniform tensorARM<int8_t, 4> firstInput;
layout(set = 0, binding = 4) readonly uniform tensorARM<int8_t, 4> secondInput;
layout(set = 0, binding = 5) writeonly uniform tensorARM<int8_t, 4> outputData;

void main() {
    uint coords[4] = uint[](0, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
    int8_t firstValue;
    int8_t secondValue;
    tensorReadARM(firstInput, coords, firstValue);
    tensorReadARM(secondInput, coords, secondValue);
    tensorWriteARM(outputData, coords, int8_t(firstValue - secondValue));
}
)";
    description.entryPoint = "main";
    description.dispatch = dispatchForNhwcTensor(shape);
    description.resources = {
        {"first_input", 0, 1, ResourceAccess::Read, tensorRequirements(vk::Format::eR8Sint, shape)},
        {"second_input", 0, 4, ResourceAccess::Read, tensorRequirements(vk::Format::eR8Sint, shape)},
        {"output", 0, 5, ResourceAccess::Write, tensorRequirements(vk::Format::eR8Sint, std::move(shape))},
    };
    return description;
}

inline DataGraphDescription makeMaxpool8x8To4x4Description() {
    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleMaxpool8x8To4x4Spirv("standalone_maxpool_8x8_to_4x4", {0, 0, 1, 1});
    description.entryPoint = "main";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, tensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16})},
        {"output", 1, 1, ResourceAccess::Write, tensorRequirements(vk::Format::eR8Sint, {1, 4, 4, 16})},
    };
    return description;
}

} // namespace mlsdk::workloadlib::test
