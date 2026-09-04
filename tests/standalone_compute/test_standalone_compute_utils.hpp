/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_utils.hpp"

#include "mlworkloadlib/workload.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace mlworkloadlib::test {

template <typename T> inline std::vector<uint8_t> bytesOf(const T &value) {
    std::vector<uint8_t> bytes(sizeof(T));
    const auto *firstByte = reinterpret_cast<const uint8_t *>(&value);
    std::copy(firstByte, firstByte + sizeof(T), bytes.begin());
    return bytes;
}

inline ComputeShaderDescription makeAddInt32BuffersDescription() {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    ComputeShaderDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleAddInt32BuffersSpirv();
    description.entryPoint = "main";
    description.dispatch = {10, 1, 1};
    description.resources = {
        {"lhs", 0, 0, ResourceAccess::Read, makeBufferRequirements(bufferSize)},
        {"rhs", 0, 1, ResourceAccess::Read, makeBufferRequirements(bufferSize)},
        {"output", 1, 2, ResourceAccess::Write, makeBufferRequirements(bufferSize)},
    };
    return description;
}

inline ComputeShaderDescription makeGlslAddInt32BuffersDescription() {
    auto description = makeAddInt32BuffersDescription();
    description.module.codeKind = ModuleCodeKind::Glsl;
    description.module.spirv.clear();
    description.module.source = R"(
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer FirstInput {
    int firstInput[];
};
layout(set = 0, binding = 1) readonly buffer SecondInput {
    int secondInput[];
};
layout(set = 1, binding = 2) writeonly buffer Output {
    int outputData[];
};
void main() {
    uint index = gl_GlobalInvocationID.x;
    outputData[index] = firstInput[index] + secondInput[index];
}
)";
    return description;
}

inline ComputeShaderDescription makeHlslAddInt32BuffersDescription() {
    auto description = makeAddInt32BuffersDescription();
    description.module.codeKind = ModuleCodeKind::Hlsl;
    description.module.spirv.clear();
    description.module.source = R"(
RWStructuredBuffer<int> firstInput : register(u0, space0);
RWStructuredBuffer<int> secondInput : register(u1, space0);
RWStructuredBuffer<int> outputData : register(u2, space1);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outputData[id.x] = firstInput[id.x] + secondInput[id.x];
}
)";
    return description;
}

inline ComputeShaderDescription makeAddInt32BuffersGlslDescription(int32_t specializationAdd) {
    auto description = makeAddInt32BuffersDescription();
    description.module = {};
    description.module.codeKind = ModuleCodeKind::Glsl;
    description.module.source = R"(
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const int spec_add = 0;

layout(push_constant) uniform PushConstants {
    int push_add;
} pc;

layout(set = 0, binding = 0) readonly buffer Lhs {
    int lhs[];
};

layout(set = 0, binding = 1) readonly buffer Rhs {
    int rhs[];
};

layout(set = 1, binding = 2) writeonly buffer Output {
    int output_values[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    output_values[index] = lhs[index] + rhs[index] + pc.push_add + spec_add + OPTION_ADD + 5;
}
    )";
    description.module.buildOptions = "-DOPTION_ADD=3";
    description.pushConstantSize = sizeof(int32_t);
    description.specializationInfo.mapEntries = {vk::SpecializationMapEntry(0, 0, sizeof(int32_t))};
    description.specializationInfo.data = bytesOf(specializationAdd);
    return description;
}

inline ResourceRequirements makeSampledImageRequirements(bool runtimeSampler) {
    ResourceRequirements resource;
    resource.kind = ResourceKind::Image;
    resource.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    resource.format = vk::Format::eR8G8B8A8Snorm;
    resource.image.extent = vk::Extent3D(2, 2, 1);
    resource.image.usage = vk::ImageUsageFlagBits::eSampled;
    resource.image.requiredLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    resource.image.range = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    if (runtimeSampler) {
        resource.image.runtimeSampler = SamplerRequirements{};
    }
    return resource;
}

} // namespace mlworkloadlib::test
