/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_utils.hpp"
#include "vgf/test_vgf_utils.hpp"

#include "mlworkloadlib/workload.hpp"

#include "vgf/encoder.hpp"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mlworkloadlib::test {

inline const std::vector<mlsdk::vgflib::GraphConstantBindingRef> noGraphConstants;

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
        {"first_input", 0, 0, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, shape)},
        {"second_input", 0, 2, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, shape)},
        {"output", 0, 3, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR8Sint, std::move(shape))},
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
        {"first_input", 0, 1, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, shape)},
        {"second_input", 0, 4, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, shape)},
        {"output", 0, 5, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR8Sint, std::move(shape))},
    };
    return description;
}

inline DataGraphDescription makeMaxpool8x8To4x4Description() {
    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleMaxpool8x8To4x4Spirv("standalone_maxpool_8x8_to_4x4", {0, 0, 1, 1});
    description.entryPoint = "main";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16})},
        {"output", 1, 1, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR8Sint, {1, 4, 4, 16})},
    };
    return description;
}

inline std::string makeMaxpool16x16To8x8Vgf() {
    const auto code = assembleMaxpool16x16To8x8Spirv("composed_vgf_maxpool_16x16_to_8x8", {0, 0, 1, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_16x16_to_8x8", "main", code);
        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "maxpool_graph_segment", {inputSet, outputSet}, {inputBinding}, {outputBinding},
                               noGraphConstants);
    });
}

inline std::string makeMaxpool8x8To4x4Vgf() {
    const auto code = assembleMaxpool8x8To4x4Spirv("composed_vgf_maxpool_8x8_to_4x4", {0, 0, 1, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4", "main", code);
        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "maxpool_graph_segment", {inputSet, outputSet}, {inputBinding}, {outputBinding},
                               noGraphConstants);
    });
}

inline std::string makeTwoSegmentMaxpoolVgf() {
    const auto firstCode = assembleMaxpool16x16To8x8Spirv("composed_vgf_first_maxpool", {0, 0, 0, 1});
    const auto secondCode = assembleMaxpool8x8To4x4Spirv("composed_vgf_second_maxpool", {0, 0, 0, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto firstModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_16x16_to_8x8", "main", firstCode);
        const auto secondModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4", "main", secondCode);

        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto intermediate =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {});

        const auto firstInputBinding = encoder.AddBindingSlot(0, input);
        const auto firstOutputBinding = encoder.AddBindingSlot(1, intermediate);
        const auto firstDescriptorSet = encoder.AddDescriptorSetInfo({firstInputBinding, firstOutputBinding}, 0);
        encoder.AddSegmentInfo(firstModule, "first_graph_segment", {firstDescriptorSet}, {firstInputBinding},
                               {firstOutputBinding}, noGraphConstants);

        const auto secondInputBinding = encoder.AddBindingSlot(0, intermediate);
        const auto secondOutputBinding = encoder.AddBindingSlot(1, output);
        const auto secondDescriptorSet = encoder.AddDescriptorSetInfo({secondInputBinding, secondOutputBinding}, 0);
        encoder.AddSegmentInfo(secondModule, "second_graph_segment", {secondDescriptorSet}, {secondInputBinding},
                               {secondOutputBinding}, noGraphConstants);
    });
}

inline std::string makeAddInt32BuffersVgf() {
    const auto code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);
        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "add_int32_buffers_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {outputBinding}, noGraphConstants, {10, 1, 1});
    });
}

} // namespace mlworkloadlib::test
