/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"
#include "test_vgf_utils.hpp"

#include "vgf/decoder.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using mlsdk::vgflib::GraphConstantBindingRef;
using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

const std::vector<GraphConstantBindingRef> noGraphConstants;

class VgfSessionExecutionTest : public RuntimeSessionExecutionTest {};

std::string makeMaxpoolVgf() {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 1, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool", "main", code);
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

std::string makeConv2dRescaleConstantVgf(const std::vector<int8_t> &weights) {
    const auto &code = assembleConv2dRescaleConstantSpirv("conv2d_rescale_constant", {0, 0, 1, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "conv2d_rescale_constant", "main", code);
        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto weightsResource = encoder.AddConstantResource(VK_FORMAT_R8_SINT, {16, 2, 2, 16}, {});
        const auto weightsConstant =
            encoder.AddConstant(weightsResource, weights.data(), weights.size() * sizeof(int8_t), -1);

        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        const std::vector<GraphConstantBindingRef> constantBindings = {
            {weightsConstant.reference, weightsConstant},
        };
        encoder.AddSegmentInfo(module, "conv2d_rescale_constant_segment", {inputSet, outputSet}, {inputBinding},
                               {outputBinding}, constantBindings);
    });
}

void fillFirstVgfConstantPayload(std::string &bytes, int8_t value) {
    char *data = bytes.data();
    auto header = mlsdk::vgflib::CreateHeaderDecoder(data, mlsdk::vgflib::HeaderSize(), bytes.size());
    if (!header) {
        throw std::runtime_error("Failed to decode VGF header");
    }

    const auto *base = reinterpret_cast<const std::byte *>(data);
    auto constants =
        mlsdk::vgflib::CreateConstantDecoder(base + header->GetConstantsOffset(), header->GetConstantsSize());
    if (!constants || constants->size() == 0) {
        throw std::runtime_error("VGF has no constants");
    }

    const auto payload = constants->getConstant(0);
    const auto *payloadBegin = reinterpret_cast<const char *>(payload.data());
    const auto offset = static_cast<std::ptrdiff_t>(payloadBegin - data);
    if (offset < 0 || static_cast<std::size_t>(offset) + payload.size() > bytes.size()) {
        throw std::runtime_error("VGF constant payload is outside the source buffer");
    }

    std::fill_n(data + offset, payload.size(), static_cast<char>(value));
}

std::string makeTwoSegmentMaxpoolVgf() {
    const auto &firstCode = assembleMaxpool16x16To8x8Spirv("maxpool_16x16_to_8x8", {0, 0, 0, 1});
    const auto &secondCode = assembleMaxpool8x8To4x4Spirv("maxpool_8x8_to_4x4", {0, 0, 0, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto firstModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_16x16_to_8x8", "main", firstCode);
        const auto secondModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4", "main", secondCode);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto firstOutput =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto firstOutputBinding = encoder.AddBindingSlot(1, firstOutput);
        const auto firstDescriptorSet = encoder.AddDescriptorSetInfo({firstInputBinding, firstOutputBinding});
        encoder.AddSegmentInfo(firstModule, "first_graph_segment", {firstDescriptorSet}, {firstInputBinding},
                               {firstOutputBinding}, noGraphConstants);

        const auto secondOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {});
        const auto secondInputBinding = encoder.AddBindingSlot(0, firstOutput);
        const auto secondOutputBinding = encoder.AddBindingSlot(1, secondOutput);
        const auto secondDescriptorSet = encoder.AddDescriptorSetInfo({secondInputBinding, secondOutputBinding});
        encoder.AddSegmentInfo(secondModule, "second_graph_segment", {secondDescriptorSet}, {secondInputBinding},
                               {secondOutputBinding}, noGraphConstants);
    });
}

std::string makeOutputBufferAliasedToIntermediateTensorVgf() {
    const auto &maxpoolCode = assembleMaxpool8x8To4x4Spirv("maxpool_8x8_to_4x4", {0, 0, 0, 1});
    const auto &addCode = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 17;
        const auto maxpoolModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4_mixed_alias", "main", maxpoolCode);
        const auto addModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers_mixed_alias", "main", addCode);

        const auto tensorInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto intermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT,
                                                                  {1, 4, 4, 16}, {}, aliasGroup);
        const auto aliasedOutputBuffer =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {64}, {4}, aliasGroup);
        const auto zeroInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto finalOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

        const auto tensorInputBinding = encoder.AddBindingSlot(0, tensorInput);
        const auto intermediateBinding = encoder.AddBindingSlot(1, intermediate);
        const auto tensorDescriptorSet = encoder.AddDescriptorSetInfo({tensorInputBinding, intermediateBinding}, 0);
        encoder.AddSegmentInfo(maxpoolModule, "write_intermediate_tensor_alias", {tensorDescriptorSet},
                               {tensorInputBinding}, {intermediateBinding}, noGraphConstants);

        const auto aliasedOutputBufferBinding = encoder.AddBindingSlot(0, aliasedOutputBuffer);
        const auto zeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto finalOutputBinding = encoder.AddBindingSlot(2, finalOutput);
        const auto bufferInputSet = encoder.AddDescriptorSetInfo({aliasedOutputBufferBinding, zeroInputBinding}, 0);
        const auto bufferOutputSet = encoder.AddDescriptorSetInfo({finalOutputBinding}, 1);
        encoder.AddSegmentInfo(addModule, "read_bound_output_buffer_alias", {bufferInputSet, bufferOutputSet},
                               {aliasedOutputBufferBinding, zeroInputBinding}, {finalOutputBinding}, noGraphConstants,
                               {10, 1, 1});
    });
}

std::string makeAddInt32BuffersVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
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

std::string makeGlslAddInt32BuffersVgf() {
    constexpr std::string_view source = R"(
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
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "glsl_add_int32_buffers", "main",
                                              mlsdk::vgflib::ShaderType::GLSL, std::string(source));

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
        encoder.AddSegmentInfo(module, "glsl_add_int32_buffers_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {outputBinding}, noGraphConstants, {10, 1, 1});
    });
}

std::string makeHlslAddInt32BuffersVgf() {
    constexpr std::string_view source = R"(
RWStructuredBuffer<int> firstInput : register(u0, space0);
RWStructuredBuffer<int> secondInput : register(u1, space0);
RWStructuredBuffer<int> outputData : register(u2, space1);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outputData[id.x] = firstInput[id.x] + secondInput[id.x];
}
)";
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "hlsl_add_int32_buffers", "main",
                                              mlsdk::vgflib::ShaderType::HLSL, std::string(source));

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
        encoder.AddSegmentInfo(module, "hlsl_add_int32_buffers_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {outputBinding}, noGraphConstants, {10, 1, 1});
    });
}

std::string makeGlslPushConstantAddInt32BuffersVgf() {
    const std::string source = R"(
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

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
    output_values[index] = lhs[index] + rhs[index] + pc.push_add;
}
)";

    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main",
                                              mlsdk::vgflib::ShaderType::GLSL, source);

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
        const auto pushConstantRange = encoder.AddPushConstRange(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t));
        encoder.AddSegmentInfo(module, "add_int32_buffers_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {outputBinding}, noGraphConstants, {10, 1, 1},
                               {pushConstantRange});
    });
}

std::string makeAddInt32BuffersWithExternalImageVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module =
            encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers_with_image", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto imageInput = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                         VK_FORMAT_R8G8B8A8_SNORM, {1, 2, 2, 4}, {});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto imageInputBinding = encoder.AddBindingSlot(3, imageInput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto inputSet =
            encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding, imageInputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "add_int32_buffers_with_external_image_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding, imageInputBinding}, {outputBinding},
                               noGraphConstants, {10, 1, 1});
    });
}

std::string makePlaceholderAddInt32BuffersVgf() {
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main");

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

std::string makeDisjointAliasedIntermediateBuffersVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 3;
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto thirdInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto fourthInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto zeroInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto firstOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto firstIntermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                       VK_FORMAT_R32_SINT, {10}, {4}, aliasGroup);
        const auto secondIntermediate =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        encoder.SetAliasGroup(secondIntermediate, aliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto firstIntermediateOutputBinding = encoder.AddBindingSlot(2, firstIntermediate);
        const auto firstInputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto firstOutputSet = encoder.AddDescriptorSetInfo({firstIntermediateOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "write_first_alias", {firstInputSet, firstOutputSet},
                               {firstInputBinding, secondInputBinding}, {firstIntermediateOutputBinding},
                               noGraphConstants, {10, 1, 1});

        const auto firstIntermediateInputBinding = encoder.AddBindingSlot(0, firstIntermediate);
        const auto zeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto firstOutputBinding = encoder.AddBindingSlot(2, firstOutput);
        const auto secondInputSet = encoder.AddDescriptorSetInfo({firstIntermediateInputBinding, zeroInputBinding}, 0);
        const auto secondOutputSet = encoder.AddDescriptorSetInfo({firstOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_first_alias", {secondInputSet, secondOutputSet},
                               {firstIntermediateInputBinding, zeroInputBinding}, {firstOutputBinding},
                               noGraphConstants, {10, 1, 1});

        const auto thirdInputBinding = encoder.AddBindingSlot(0, thirdInput);
        const auto fourthInputBinding = encoder.AddBindingSlot(1, fourthInput);
        const auto secondIntermediateOutputBinding = encoder.AddBindingSlot(2, secondIntermediate);
        const auto thirdInputSet = encoder.AddDescriptorSetInfo({thirdInputBinding, fourthInputBinding}, 0);
        const auto thirdOutputSet = encoder.AddDescriptorSetInfo({secondIntermediateOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "write_second_alias", {thirdInputSet, thirdOutputSet},
                               {thirdInputBinding, fourthInputBinding}, {secondIntermediateOutputBinding},
                               noGraphConstants, {10, 1, 1});

        const auto secondIntermediateInputBinding = encoder.AddBindingSlot(0, secondIntermediate);
        const auto repeatedZeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto secondOutputBinding = encoder.AddBindingSlot(2, secondOutput);
        const auto fourthInputSet =
            encoder.AddDescriptorSetInfo({secondIntermediateInputBinding, repeatedZeroInputBinding}, 0);
        const auto fourthOutputSet = encoder.AddDescriptorSetInfo({secondOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_second_alias", {fourthInputSet, fourthOutputSet},
                               {secondIntermediateInputBinding, repeatedZeroInputBinding}, {secondOutputBinding},
                               noGraphConstants, {10, 1, 1});
    });
}

std::string makeIndependentAliasGroupsVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t firstAliasGroup = 5;
        constexpr uint32_t secondAliasGroup = 7;
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto thirdInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto fourthInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto firstIntermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                       VK_FORMAT_R32_SINT, {10}, {4}, firstAliasGroup);
        const auto secondIntermediate = encoder.AddIntermediateResource(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4}, secondAliasGroup);
        const auto firstAliasedOutput = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT,
                                                                  {10}, {4}, firstAliasGroup);
        const auto secondAliasedOutput = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                   VK_FORMAT_R32_SINT, {10}, {4}, secondAliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto firstIntermediateBinding = encoder.AddBindingSlot(2, firstIntermediate);
        const auto firstInputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto firstOutputSet = encoder.AddDescriptorSetInfo({firstIntermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_first_alias_group", {firstInputSet, firstOutputSet},
                               {firstInputBinding, secondInputBinding}, {firstIntermediateBinding}, noGraphConstants,
                               {10, 1, 1});

        const auto thirdInputBinding = encoder.AddBindingSlot(0, thirdInput);
        const auto fourthInputBinding = encoder.AddBindingSlot(1, fourthInput);
        const auto secondIntermediateBinding = encoder.AddBindingSlot(2, secondIntermediate);
        const auto secondInputSet = encoder.AddDescriptorSetInfo({thirdInputBinding, fourthInputBinding}, 0);
        const auto secondOutputSet = encoder.AddDescriptorSetInfo({secondIntermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_second_alias_group", {secondInputSet, secondOutputSet},
                               {thirdInputBinding, fourthInputBinding}, {secondIntermediateBinding}, noGraphConstants,
                               {10, 1, 1});

        const auto firstAliasedOutputBinding = encoder.AddBindingSlot(0, firstAliasedOutput);
        const auto secondAliasedOutputBinding = encoder.AddBindingSlot(1, secondAliasedOutput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto thirdInputSet =
            encoder.AddDescriptorSetInfo({firstAliasedOutputBinding, secondAliasedOutputBinding}, 0);
        const auto thirdOutputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_both_alias_groups", {thirdInputSet, thirdOutputSet},
                               {firstAliasedOutputBinding, secondAliasedOutputBinding}, {outputBinding},
                               noGraphConstants, {10, 1, 1});
    });
}

std::string makeOutputAliasedToIntermediateBufferVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 11;
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto zeroInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto finalOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto intermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT,
                                                                  {10}, {4}, aliasGroup);
        const auto aliasedOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4}, aliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto intermediateBinding = encoder.AddBindingSlot(2, intermediate);
        const auto firstInputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto firstOutputSet = encoder.AddDescriptorSetInfo({intermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_intermediate_alias", {firstInputSet, firstOutputSet},
                               {firstInputBinding, secondInputBinding}, {intermediateBinding}, noGraphConstants,
                               {10, 1, 1});

        const auto aliasedOutputInputBinding = encoder.AddBindingSlot(0, aliasedOutput);
        const auto zeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto finalOutputBinding = encoder.AddBindingSlot(2, finalOutput);
        const auto secondInputSet = encoder.AddDescriptorSetInfo({aliasedOutputInputBinding, zeroInputBinding}, 0);
        const auto secondOutputSet = encoder.AddDescriptorSetInfo({finalOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_output_alias", {secondInputSet, secondOutputSet},
                               {aliasedOutputInputBinding, zeroInputBinding}, {finalOutputBinding}, noGraphConstants,
                               {10, 1, 1});
    });
}

std::string makeOutputTensorAliasedToIntermediateBufferVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 23;
        const auto module =
            encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers_tensor_alias", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto aliasedOutputTensor =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {40}, {}, aliasGroup);
        const auto intermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT,
                                                                  {10}, {4}, aliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto intermediateBinding = encoder.AddBindingSlot(2, intermediate);
        const auto inputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({intermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_buffer_alias_of_output_tensor", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {intermediateBinding}, noGraphConstants,
                               {10, 1, 1});
        (void)aliasedOutputTensor;
    });
}

std::string makeOutputTensorAliasedToIntermediateTensorVgf() {
    const auto &code = assembleMaxpool8x8To4x4Spirv("maxpool_8x8_to_4x4_tensor_alias", {0, 0, 0, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 31;
        const auto module =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4_tensor_alias", "main", code);

        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto aliasedOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {}, aliasGroup);
        const auto intermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT,
                                                                  {1, 4, 4, 16}, {}, aliasGroup);

        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto intermediateBinding = encoder.AddBindingSlot(1, intermediate);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, intermediateBinding}, 0);
        encoder.AddSegmentInfo(module, "write_tensor_alias_of_output_tensor", {descriptorSet}, {inputBinding},
                               {intermediateBinding}, noGraphConstants);
        (void)aliasedOutput;
    });
}

std::string makeOutputImageAliasedToIntermediateImageVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 37;
        const auto module =
            encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers_image_alias", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto aliasedOutputImage = encoder.AddOutputResource(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_FORMAT_R8G8B8A8_SNORM, {1, 2, 3, 4}, {}, aliasGroup);
        const auto intermediateImage = encoder.AddIntermediateResource(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_FORMAT_R8G8B8A8_SNORM, {1, 2, 3, 4}, {}, aliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto intermediateImageBinding = encoder.AddBindingSlot(3, intermediateImage);
        const auto inputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding, intermediateImageBinding}, 1);
        encoder.AddSegmentInfo(module, "write_image_alias_metadata", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {outputBinding, intermediateImageBinding},
                               noGraphConstants, {10, 1, 1});
        (void)aliasedOutputImage;
    });
}

std::string makeAliasVgf(AliasScenario scenario) {
    switch (scenario) {
    case AliasScenario::OutputBufferAliasedToIntermediateBuffer:
        return makeOutputAliasedToIntermediateBufferVgf();
    case AliasScenario::OutputBufferAliasedToIntermediateTensor:
        return makeOutputBufferAliasedToIntermediateTensorVgf();
    case AliasScenario::OutputTensorAliasedToIntermediateBuffer:
        return makeOutputTensorAliasedToIntermediateBufferVgf();
    case AliasScenario::OutputTensorAliasedToIntermediateTensor:
        return makeOutputTensorAliasedToIntermediateTensorVgf();
    case AliasScenario::OutputImageAliasedToIntermediateImage:
        return makeOutputImageAliasedToIntermediateImageVgf();
    }
    throw std::logic_error("Unhandled alias scenario");
}

} // namespace

class VgfAliasExecutionTest : public AliasExecutionTestBase, public testing::WithParamInterface<AliasCase> {};

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST_P(VgfAliasExecutionTest, Run) {
    const auto bytes = makeAliasVgf(GetParam().scenario);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    runAliasCase(workload, GetParam(), true);
}

TEST_F(VgfSessionExecutionTest, RunComputeShaderSegment) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RunGlslComputeShaderSegment) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled in this build";
    }

    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeGlslAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    const std::vector<int32_t> secondInput = {10, -9, 8, -7, 6, -5, 4, -3, 2, -1};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RunHlslComputeShaderSegment) {
    if (!supports(Feature::HlslModules)) {
        GTEST_SKIP() << "HLSL source module support is not enabled in this build";
    }

    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeHlslAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {10, 9, 8, 7, 6, -5, -4, -3, -2, -1};
    const std::vector<int32_t> secondInput = {-1, -2, -3, -4, -5, 6, 7, 8, 9, 10};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RunComputeShaderSegmentWithBoundPlaceholderModule) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makePlaceholderAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    const std::vector<int32_t> secondInput = {5, 8, 9, 7, 9, 3, 2, 3, 8, 4};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    ASSERT_EQ(workload.placeholderModuleCount(), 1);
    ModuleImplementation implementation;
    implementation.codeKind = ModuleCodeKind::Spirv;
    implementation.spirv = assembleAddInt32BuffersSpirv();

    Session session(context, workload);
    session.bindModule(workload.placeholderModule(0), std::move(implementation));
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RunGlslComputeShaderSegmentWithPushConstants) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled in this build";
    }

    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);
    constexpr int32_t pushAdd = 13;

    const auto bytes = makeGlslPushConstantAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    const std::vector<int32_t> secondInput = {10, 20, -30, -40, 50, 60, -70, -80, 90, 100};
    auto expected = addVectors(firstInput, secondInput);
    for (auto &value : expected) {
        value += pushAdd;
    }

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});
    bindings.bindPushConstants(&pushAdd, sizeof(pushAdd));

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RecordComputeShaderSegment) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {2, 4, 6, 8, 10, -12, -14, 16, 18, 20};
    const std::vector<int32_t> secondInput = {3, -3, 5, -5, 7, -7, 11, -11, 13, -13};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    auto commandBuffer =
        std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    execution.record(*commandBuffer);
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RunComputeShaderSegmentWithExternalImageBinding) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeAddInt32BuffersWithExternalImageVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);
    const Image imageInput(physicalDevice, device);

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 4);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    const vk::SamplerCreateInfo samplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                                                  vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
                                                  vk::SamplerAddressMode::eClampToEdge,
                                                  vk::SamplerAddressMode::eClampToEdge);
    const vk::raii::Sampler sampler(device, samplerCreateInfo);
    const auto imageRequirements = workload.resource(2).requirements().asImage();
    ImageBindingInfo imageBindingInfo;
    imageBindingInfo.image = *imageInput.image;
    imageBindingInfo.imageView = *imageInput.imageView;
    imageBindingInfo.sampler = *sampler;
    imageBindingInfo.subresourceRange = imageRequirements.requiredSubresourceRange();
    bindings.bindImage(workload.resource(2), imageBindingInfo);
    bindings.bindBuffer(workload.resource(3), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);

    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, RunDisjointAliasedIntermediateBuffers) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeDisjointAliasedIntermediateBuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    ASSERT_EQ(workload.executableCount(), 4);

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer thirdInputBuffer(physicalDevice, device, bufferSize);
    const Buffer fourthInputBuffer(physicalDevice, device, bufferSize);
    const Buffer zeroInputBuffer(physicalDevice, device, bufferSize);
    const Buffer firstOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondOutputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    const std::vector<int32_t> secondInput = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    const std::vector<int32_t> thirdInput = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    const std::vector<int32_t> fourthInput = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const std::vector<int32_t> zeros(elements, 0);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    thirdInputBuffer.write(thirdInput);
    fourthInputBuffer.write(fourthInput);
    zeroInputBuffer.write(zeros);
    firstOutputBuffer.write(zeros);
    secondOutputBuffer.write(zeros);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 7);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*thirdInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(3), BufferBindingInfo{*fourthInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(4), BufferBindingInfo{*zeroInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(5), BufferBindingInfo{*firstOutputBuffer.buffer});
    bindings.bindBuffer(workload.resource(6), BufferBindingInfo{*secondOutputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(firstOutputBuffer.read(elements), addVectors(firstInput, secondInput));
    EXPECT_EQ(secondOutputBuffer.read(elements), addVectors(thirdInput, fourthInput));
}

TEST_F(VgfSessionExecutionTest, RunIndependentIntermediateAliasGroups) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeIndependentAliasGroupsVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    ASSERT_EQ(workload.executableCount(), 3);

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer thirdInputBuffer(physicalDevice, device, bufferSize);
    const Buffer fourthInputBuffer(physicalDevice, device, bufferSize);
    const Buffer firstAliasedOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondAliasedOutputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55};
    const std::vector<int32_t> secondInput = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
    const std::vector<int32_t> thirdInput = {55, 34, 21, 13, 8, 5, 3, 2, 1, 1};
    const std::vector<int32_t> fourthInput = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const std::vector<int32_t> zeros(elements, 0);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    thirdInputBuffer.write(thirdInput);
    fourthInputBuffer.write(fourthInput);
    firstAliasedOutputBuffer.write(zeros);
    secondAliasedOutputBuffer.write(zeros);
    outputBuffer.write(zeros);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 7);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*thirdInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(3), BufferBindingInfo{*fourthInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(4), BufferBindingInfo{*outputBuffer.buffer});
    bindings.bindBuffer(workload.resource(5),
                        BufferBindingInfo{*firstAliasedOutputBuffer.buffer,
                                          {*firstAliasedOutputBuffer.memory, 0, firstAliasedOutputBuffer.memorySize}});
    bindings.bindBuffer(workload.resource(6), BufferBindingInfo{*secondAliasedOutputBuffer.buffer,
                                                                {*secondAliasedOutputBuffer.memory, 0,
                                                                 secondAliasedOutputBuffer.memorySize}});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements),
              addVectors(addVectors(firstInput, secondInput), addVectors(thirdInput, fourthInput)));
}

TEST_F(VgfSessionExecutionTest, RunMaxpoolDataVgf) {
    const auto bytes = makeMaxpoolVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});

    const auto input = makeMaxpoolInput(inputTensor.shape);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});
    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
}

TEST_F(VgfSessionExecutionTest, RunGraphWithConstant) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};

    const std::vector<int8_t> zeroWeights(16UL * 2 * 2 * 16, 0);
    const std::vector<int8_t> oneWeights(16UL * 2 * 2 * 16, 1);
    const auto zeroWeightVgf = makeConv2dRescaleConstantVgf(zeroWeights);
    const auto oneWeightVgf = makeConv2dRescaleConstantVgf(oneWeights);
    auto zeroWeightWorkload = Workload::fromVGF(zeroWeightVgf.data(), zeroWeightVgf.size());
    auto oneWeightWorkload = Workload::fromVGF(oneWeightVgf.data(), oneWeightVgf.size());
    auto context = wrappedContext();

    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 7);
    inputTensor.write(input);

    auto run = [&](Workload &workload) {
        outputTensor.fill(0, outputTensor.numElements());

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        EXPECT_EQ(workload.resourceCount(), 2);
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

        auto execution = session.prepare(bindings);
        execution.run();
        return outputTensor.read(outputTensor.numElements());
    };

    const auto zeroWeightOutput = run(zeroWeightWorkload);
    const auto oneWeightOutput = run(oneWeightWorkload);

    EXPECT_NE(zeroWeightOutput, oneWeightOutput);
}

TEST_F(VgfSessionExecutionTest, RunMemoryVgfBorrowsConstantPayload) {
    const std::vector<int64_t> inputShape = {1, 16, 16, 16};
    const std::vector<int64_t> outputShape = {1, 8, 8, 16};
    constexpr std::size_t weightCount = 16UL * 2 * 2 * 16;

    const std::vector<int8_t> zeroWeights(weightCount, 0);
    const std::vector<int8_t> oneWeights(weightCount, 1);
    const auto zeroWeightVgf = makeConv2dRescaleConstantVgf(zeroWeights);
    const auto oneWeightVgf = makeConv2dRescaleConstantVgf(oneWeights);
    auto borrowedVgf = makeConv2dRescaleConstantVgf(zeroWeights);
    auto zeroWeightWorkload = Workload::fromVGF(zeroWeightVgf.data(), zeroWeightVgf.size());
    auto oneWeightWorkload = Workload::fromVGF(oneWeightVgf.data(), oneWeightVgf.size());
    auto borrowedWorkload = Workload::fromVGF(borrowedVgf.data(), borrowedVgf.size());
    fillFirstVgfConstantPayload(borrowedVgf, 1);

    auto context = wrappedContext();
    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, inputShape);
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, outputShape);

    const auto input = makeMaxpoolInput(inputShape, 7);
    inputTensor.write(input);

    auto run = [&](Workload &workload) {
        outputTensor.fill(0, outputTensor.numElements());

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        EXPECT_EQ(workload.resourceCount(), 2);
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});

        auto execution = session.prepare(bindings);
        execution.run();
        return outputTensor.read(outputTensor.numElements());
    };

    const auto borrowedOutput = run(borrowedWorkload);
    EXPECT_EQ(borrowedOutput, run(oneWeightWorkload));
    EXPECT_NE(borrowedOutput, run(zeroWeightWorkload));
}

TEST_F(VgfSessionExecutionTest, RunMaxpoolFileVgf) {
    const auto bytes = makeMaxpoolVgf();
    const TempFolder tempFolder("mlworkloadlib_vgf_session");
    const auto path = tempFolder.relative("full_maxpool.vgf");
    {
        std::ofstream file(path, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    auto workload = Workload::fromVGF(path);
    auto context = wrappedContext();

    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});

    const auto input = makeMaxpoolInput(inputTensor.shape, 0);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});
    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
}

TEST_F(VgfSessionExecutionTest, RunMaxpoolRepeatedDifferentInput) {
    const auto bytes = makeMaxpoolVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});
    auto execution = session.prepare(bindings);

    for (const auto seed : {3U, 41U, 67U, 89U, 113U}) {
        const auto input = makeMaxpoolInput(inputTensor.shape, seed);
        inputTensor.write(input);
        outputTensor.fill(0, outputTensor.numElements());
        execution.run();
        EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
    }
}

TEST_F(VgfSessionExecutionTest, RunTwoMaxpoolGraphSegments) {
    const auto bytes = makeTwoSegmentMaxpoolVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    ASSERT_EQ(workload.executableCount(), 2);

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor secondOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto firstInput = makeMaxpoolInput(firstInputTensor.shape, 7);
    firstInputTensor.write(firstInput);
    secondOutputTensor.fill(0, secondOutputTensor.numElements());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 2);
    bindings.bindTensor(workload.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    bindings.bindTensor(workload.resource(1), TensorBindingInfo{*secondOutputTensor.tensor});

    auto execution = session.prepare(bindings);
    execution.run();

    const auto firstExpected = expectedMaxpool(firstInput, firstInputTensor.shape);
    EXPECT_EQ(secondOutputTensor.read(secondOutputTensor.numElements()), expectedMaxpool(firstExpected, {1, 8, 8, 16}));
}

/*******************************************************************************
 * Validation coverage
 *******************************************************************************/

TEST_F(VgfSessionExecutionTest, ConfigureFailsWhenPlaceholderModuleIsUnbound) {
    const auto bytes = makePlaceholderAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});

    ASSERT_EQ(workload.placeholderModuleCount(), 1);
    EXPECT_TRUE(workload.placeholderModule(0).module().requiresImplementation());
    uint32_t placeholderCount = 0;
    for (const auto placeholderModule : workload.placeholderModules()) {
        EXPECT_EQ(placeholderModule.index(), placeholderCount);
        EXPECT_TRUE(placeholderModule.module().requiresImplementation());
        ++placeholderCount;
    }
    EXPECT_EQ(placeholderCount, workload.placeholderModuleCount());

    Session session(context, workload);
    EXPECT_THROW(session.configure(), std::runtime_error);
}

TEST_P(VgfAliasExecutionTest, PrepareRequiresBoundMemoryInfo) {
    const auto bytes = makeAliasVgf(GetParam().scenario);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    runAliasCase(workload, GetParam(), false);
}

INSTANTIATE_TEST_SUITE_P(AliasCases, VgfAliasExecutionTest,
                         testing::Values(AliasCase{"OutputBufferAliasedToIntermediateBuffer",
                                                   AliasScenario::OutputBufferAliasedToIntermediateBuffer, 2, 5},
                                         AliasCase{"OutputBufferAliasedToIntermediateTensor",
                                                   AliasScenario::OutputBufferAliasedToIntermediateTensor, 2, 4},
                                         AliasCase{"OutputTensorAliasedToIntermediateBuffer",
                                                   AliasScenario::OutputTensorAliasedToIntermediateBuffer, 1, 3},
                                         AliasCase{"OutputTensorAliasedToIntermediateTensor",
                                                   AliasScenario::OutputTensorAliasedToIntermediateTensor, 1, 2},
                                         AliasCase{"OutputImageAliasedToIntermediateImage",
                                                   AliasScenario::OutputImageAliasedToIntermediateImage, 1, 4}),
                         aliasCaseName);
