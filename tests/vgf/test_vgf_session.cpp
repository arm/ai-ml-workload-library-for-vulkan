/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"
#include "mlworkloadlib_utils/application_context.hpp"
#include "test_alias_execution.hpp"
#include "test_expected_results.hpp"
#include "test_resource_requirements.hpp"
#include "test_spirv_utils.hpp"
#include "test_vgf_utils.hpp"
#include "test_vulkan_fixture.hpp"
#include "test_vulkan_resources.hpp"

#include "vgf-utils/temp_folder.hpp"
#include "vgf/decoder.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using mlsdk::vgflib::GraphConstantBindingRef;
using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::test;

class VgfSessionExecutionTest : public RuntimeSessionExecutionTest {};

// Capture library arguments before forwarding to Vulkan. The emulation layer
// can accept mismatched tensor tilings, so execution alone is insufficient.
class ResourceCreationRecorder {
  public:
    explicit ResourceCreationRecorder(vk::raii::Device &device)
        : createTensor_(device, &vk::raii::detail::DeviceDispatcher::vkCreateTensorARM, createTensor),
          createImage_(device, &vk::raii::detail::DeviceDispatcher::vkCreateImage, createImage),
          createPipelines_(device, &vk::raii::detail::DeviceDispatcher::vkCreateDataGraphPipelinesARM, createPipelines),
          previous_(active_) {
        active_ = this;
    }

    ~ResourceCreationRecorder() { active_ = previous_; }

    ResourceCreationRecorder(const ResourceCreationRecorder &) = delete;
    ResourceCreationRecorder &operator=(const ResourceCreationRecorder &) = delete;
    ResourceCreationRecorder(ResourceCreationRecorder &&) = delete;
    ResourceCreationRecorder &operator=(ResourceCreationRecorder &&) = delete;

    // Vulkan callbacks append observations even when the hook guard is const.
    mutable std::vector<vk::TensorTilingARM> tensorTilings;
    mutable std::vector<vk::ImageTiling> imageTilings;
    mutable std::vector<vk::Format> imageFormats;
    mutable std::vector<vk::ImageUsageFlags> imageUsages;
    mutable std::vector<vk::TensorTilingARM> pipelineTensorTilings;
    mutable std::vector<vk::Format> pipelineTensorFormats;
    mutable std::vector<std::vector<int64_t>> pipelineTensorShapes;
    mutable std::vector<bool> pipelineTensorHasStrides;
    mutable std::vector<std::vector<vk::StructureType>> pipelineResourceChains;
    mutable std::vector<vk::ImageLayout> pipelineImageLayouts;
    bool forwardPipelineCreation = true;

  private:
    static VKAPI_ATTR VkResult VKAPI_CALL createTensor(VkDevice device, const VkTensorCreateInfoARM *createInfo,
                                                       const VkAllocationCallbacks *allocator, VkTensorARM *tensor) {
        active_->tensorTilings.push_back(vk::TensorTilingARM(createInfo->pDescription->tiling));
        return active_->createTensor_.original()(device, createInfo, allocator, tensor);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createImage(VkDevice device, const VkImageCreateInfo *createInfo,
                                                      const VkAllocationCallbacks *allocator, VkImage *image) {
        active_->imageTilings.push_back(vk::ImageTiling(createInfo->tiling));
        active_->imageFormats.push_back(vk::Format(createInfo->format));
        active_->imageUsages.emplace_back(createInfo->usage);
        return active_->createImage_.original()(device, createInfo, allocator, image);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createPipelines(VkDevice device, VkDeferredOperationKHR deferredOperation,
                                                          VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                          const VkDataGraphPipelineCreateInfoARM *createInfos,
                                                          const VkAllocationCallbacks *allocator,
                                                          VkPipeline *pipelines) {
        for (uint32_t index = 0; index < createInfoCount; ++index) {
            const auto &createInfo = createInfos[index];
            for (uint32_t resourceIndex = 0; resourceIndex < createInfo.resourceInfoCount; ++resourceIndex) {
                const auto *next =
                    static_cast<const VkBaseInStructure *>(createInfo.pResourceInfos[resourceIndex].pNext);
                active_->pipelineResourceChains.emplace_back();
                while (next != nullptr) {
                    active_->pipelineResourceChains.back().push_back(vk::StructureType(next->sType));
                    if (next->sType == VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM) {
                        const auto *description = reinterpret_cast<const VkTensorDescriptionARM *>(next);
                        active_->pipelineTensorTilings.push_back(vk::TensorTilingARM(description->tiling));
                        active_->pipelineTensorFormats.push_back(vk::Format(description->format));
                        active_->pipelineTensorShapes.emplace_back(
                            description->pDimensions, description->pDimensions + description->dimensionCount);
                        active_->pipelineTensorHasStrides.push_back(description->pStrides != nullptr);
                    } else if (next->sType == VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_IMAGE_LAYOUT_ARM) {
                        const auto *layout =
                            reinterpret_cast<const VkDataGraphPipelineResourceInfoImageLayoutARM *>(next);
                        active_->pipelineImageLayouts.push_back(vk::ImageLayout(layout->layout));
                    }
                    next = next->pNext;
                }
            }
        }
        if (!active_->forwardPipelineCreation) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return active_->createPipelines_.original()(device, deferredOperation, pipelineCache, createInfoCount,
                                                    createInfos, allocator, pipelines);
    }

    ScopedDeviceDispatcherHook<PFN_vkCreateTensorARM> createTensor_;
    ScopedDeviceDispatcherHook<PFN_vkCreateImage> createImage_;
    ScopedDeviceDispatcherHook<PFN_vkCreateDataGraphPipelinesARM> createPipelines_;
    const ResourceCreationRecorder *previous_;
    inline static thread_local const ResourceCreationRecorder *active_ = nullptr;
};

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
        addInt32BuffersSegment(encoder, module, "glsl_add_int32_buffers_segment");
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
        addInt32BuffersSegment(encoder, module, "hlsl_add_int32_buffers_segment");
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
        const auto pushConstantRange = encoder.AddPushConstRange(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t));
        addInt32BuffersSegment(encoder, module, "add_int32_buffers_segment", {pushConstantRange});
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
        addInt32BuffersSegment(encoder, module, "add_int32_buffers_segment");
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

std::string makeOutputTensorAliasedToIntermediateTensorVgf(bool stridedPeer = false, bool publicPeer = false) {
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
        if (stridedPeer) {
            (void)encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16},
                                                  {256, 64, 16, 1}, aliasGroup);
        }
        if (publicPeer) {
            (void)encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {},
                                            aliasGroup);
        }

        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto intermediateBinding = encoder.AddBindingSlot(1, intermediate);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, intermediateBinding}, 0);
        encoder.AddSegmentInfo(module, "write_tensor_alias_of_output_tensor", {descriptorSet}, {inputBinding},
                               {intermediateBinding}, noGraphConstants);
        (void)aliasedOutput;
    });
}

std::string makeOutputImageAliasedToIntermediateImageVgf(
    VkDescriptorType outputDescriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VkDescriptorType intermediateDescriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
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
        const auto aliasedOutputImage =
            encoder.AddOutputResource(outputDescriptorType, VK_FORMAT_R8G8B8A8_SNORM, {1, 2, 3, 4}, {}, aliasGroup);
        const auto intermediateImage = encoder.AddIntermediateResource(
            intermediateDescriptorType, VK_FORMAT_R8G8B8A8_SNORM, {1, 2, 3, 4}, {}, aliasGroup);
        if (outputDescriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            encoder.AddSamplerConfig(aliasedOutputImage, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                     VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
        }
        if (intermediateDescriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            encoder.AddSamplerConfig(intermediateImage, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                     VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
        }

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

TEST_F(VgfAliasExecutionTest, RunOptimalTensorAliasWithSessionSelection) {
    const auto bytes = makeOutputTensorAliasedToIntermediateTensorVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const ResourceCreationRecorder recorder(device);

    runAliasCase(workload, aliasCases[3], true, vk::TensorTilingARM::eOptimal);

    EXPECT_EQ(recorder.pipelineTensorTilings,
              (std::vector<vk::TensorTilingARM>{vk::TensorTilingARM::eLinear, vk::TensorTilingARM::eOptimal}));
    EXPECT_EQ(recorder.tensorTilings,
              (std::vector<vk::TensorTilingARM>{vk::TensorTilingARM::eLinear, vk::TensorTilingARM::eOptimal,
                                                vk::TensorTilingARM::eOptimal, vk::TensorTilingARM::eLinear}));
}

TEST_F(VgfAliasExecutionTest, RunLinearImageAliasWithSessionSelection) {
    const auto bytes = makeOutputImageAliasedToIntermediateImageVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const ResourceCreationRecorder recorder(device);

    runAliasCase(workload, aliasCases[4], true, std::nullopt, vk::ImageTiling::eLinear);

    EXPECT_EQ(recorder.imageTilings,
              (std::vector<vk::ImageTiling>{vk::ImageTiling::eLinear, vk::ImageTiling::eLinear}));
    const auto aliasedUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
    EXPECT_EQ(recorder.imageUsages, (std::vector<vk::ImageUsageFlags>{aliasedUsage, aliasedUsage}));
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
    const auto expected = addVectors(firstInput, secondInput);

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
    const auto expected = addVectors(firstInput, secondInput);

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
    const auto expected = addVectors(firstInput, secondInput);

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
    const auto expected = addVectors(firstInput, secondInput);

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

TEST_F(VgfSessionExecutionTest, SharedCompiledExecutionRetriesAfterFailedConfigure) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makePlaceholderAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    CompiledExecution compiledExecution(context, workload);

    Session session(context, workload, compiledExecution);
    EXPECT_THROW(session.configure(), std::runtime_error);

    ModuleImplementation implementation;
    implementation.codeKind = ModuleCodeKind::Spirv;
    implementation.spirv = assembleAddInt32BuffersSpirv();
    session.bindModule(workload.placeholderModule(0), std::move(implementation));
    session.configure();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {6, 2, 8, 1, 8, 2, 8, 4, 5, 9};
    const std::vector<int32_t> secondInput = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), addVectors(firstInput, secondInput));
}

TEST_F(VgfSessionExecutionTest, SharedCompiledExecutionRejectsModuleBindingsAfterCompilation) {
    const auto bytes = makePlaceholderAddInt32BuffersVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    CompiledExecution compiledExecution(context, workload);
    const auto spirv = assembleAddInt32BuffersSpirv();

    const auto bindPlaceholder = [&workload](Session &session, std::vector<uint32_t> code) {
        ModuleImplementation implementation;
        implementation.codeKind = ModuleCodeKind::Spirv;
        implementation.spirv = std::move(code);
        session.bindModule(workload.placeholderModule(0), std::move(implementation));
    };

    Session preboundSession(context, workload, compiledExecution);
    bindPlaceholder(preboundSession, spirv);

    {
        Session firstSession(context, workload, compiledExecution);
        bindPlaceholder(firstSession, spirv);
        firstSession.configure();
    }

    try {
        preboundSession.configure();
        FAIL() << "Pre-bound placeholder module was silently ignored";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ("Compilation inputs cannot be rebound after compiled execution is configured", error.what());
    }

    Session unboundSession(context, workload, compiledExecution);
    EXPECT_NO_THROW(unboundSession.configure());

    Session lateBindingSession(context, workload, compiledExecution);
    try {
        bindPlaceholder(lateBindingSession, spirv);
        FAIL() << "Late placeholder module binding was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ("Compilation inputs cannot be rebound after compiled execution is configured", error.what());
    }
    EXPECT_NO_THROW(lateBindingSession.configure());
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
    bindings.bindPushConstants(workload.executable(0),
                               {reinterpret_cast<const std::byte *>(&pushAdd), sizeof(pushAdd)});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfSessionExecutionTest, IndependentPushConstantsRetainOffsets) {
    auto context = wrappedContext();
    const auto bytes = makeTwoSegmentPushConstantVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const Buffer firstOutput(physicalDevice, device, sizeof(int32_t));
    const Buffer secondOutput(physicalDevice, device, sizeof(int32_t));
    std::array<int32_t, 2> firstPayload = {-100, 23};
    const std::array<int32_t, 3> secondPayload = {-200, 57, -300};
    const auto *firstData = reinterpret_cast<const std::byte *>(firstPayload.data());
    const auto *secondData = reinterpret_cast<const std::byte *>(secondPayload.data());

    ASSERT_EQ(workload.executable(0).module().index(), workload.executable(1).module().index());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstOutput.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondOutput.buffer});

    const auto firstStep = workload.executable(0);
    const auto secondStep = workload.executable(1);

    // First step requires 8 bytes: offset 4 + size 4.
    EXPECT_THROW(bindings.bindPushConstants(firstStep, {firstData, sizeof(int32_t)}), std::runtime_error);
    // Second step requires 12 bytes: offset 4 + size 8.
    EXPECT_THROW(bindings.bindPushConstants(secondStep, {firstData, sizeof(firstPayload)}), std::runtime_error);

    bindings.bindPushConstants(firstStep, {firstData, sizeof(firstPayload)});
    EXPECT_THROW(session.prepare(bindings), std::runtime_error);
    bindings.bindPushConstants(secondStep, {secondData, sizeof(secondPayload)});

    auto execution = session.prepare(bindings);

    execution.run();
    EXPECT_EQ(firstOutput.read(1), std::vector<int32_t>{23});
    EXPECT_EQ(secondOutput.read(1), std::vector<int32_t>{57});
}

TEST_F(VgfSessionExecutionTest, PreparedExecutionsRetainPushConstantSnapshots) {
    auto context = wrappedContext();
    const auto bytes = makeTwoSegmentPushConstantVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const Buffer firstOutput(physicalDevice, device, sizeof(int32_t));
    const Buffer secondOutput(physicalDevice, device, sizeof(int32_t));
    std::array<int32_t, 2> firstPayload = {-100, 23};
    const std::array<int32_t, 3> secondPayload = {-200, 57, -300};
    const auto *firstData = reinterpret_cast<const std::byte *>(firstPayload.data());
    const auto *secondData = reinterpret_cast<const std::byte *>(secondPayload.data());

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstOutput.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondOutput.buffer});

    bindings.bindPushConstants(workload.executable(0), {firstData, sizeof(firstPayload)});
    bindings.bindPushConstants(workload.executable(1), {secondData, sizeof(secondPayload)});
    auto firstExecution = session.prepare(bindings);

    firstPayload[1] = 42;
    bindings.bindPushConstants(workload.executable(0), {firstData, sizeof(firstPayload)});
    auto secondExecution = session.prepare(bindings);

    secondExecution.run();
    EXPECT_EQ(firstOutput.read(1), std::vector<int32_t>{42});
    EXPECT_EQ(secondOutput.read(1), std::vector<int32_t>{57});

    firstExecution.run();
    EXPECT_EQ(firstOutput.read(1), std::vector<int32_t>{23});
    EXPECT_EQ(secondOutput.read(1), std::vector<int32_t>{57});
}

TEST_F(VgfSessionExecutionTest, SharedCompiledExecutionKeepsPushConstantsIndependent) {
    auto context = wrappedContext();
    const auto bytes = makeTwoSegmentPushConstantVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const Buffer firstOutput(physicalDevice, device, sizeof(int32_t));
    const Buffer secondOutput(physicalDevice, device, sizeof(int32_t));
    std::array<int32_t, 2> firstPayload = {-100, 23};
    const std::array<int32_t, 3> secondPayload = {-200, 57, -300};
    const auto *firstData = reinterpret_cast<const std::byte *>(firstPayload.data());
    const auto *secondData = reinterpret_cast<const std::byte *>(secondPayload.data());

    CompiledExecution compiled(context, workload);
    Session session(context, workload, compiled);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstOutput.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondOutput.buffer});

    const auto firstStep = workload.executable(0);
    const auto secondStep = workload.executable(1);
    bindings.bindPushConstants(firstStep, {firstData, sizeof(firstPayload)});
    bindings.bindPushConstants(secondStep, {secondData, sizeof(secondPayload)});
    auto firstExecution = session.prepare(bindings);

    Session otherSession(context, workload, compiled);
    otherSession.configure();
    auto otherBindings = otherSession.createBindingSet();
    otherBindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstOutput.buffer});
    otherBindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondOutput.buffer});

    EXPECT_TRUE(otherBindings.pushConstants(firstStep).empty());
    otherBindings.bindPushConstants(secondStep, {secondData, sizeof(secondPayload)});
    EXPECT_THROW(otherSession.prepare(otherBindings), std::runtime_error);

    firstPayload[1] = 99;
    otherBindings.bindPushConstants(firstStep, {firstData, sizeof(firstPayload)});
    auto otherExecution = otherSession.prepare(otherBindings);

    otherExecution.run();
    EXPECT_EQ(firstOutput.read(1), std::vector<int32_t>{99});
    EXPECT_EQ(secondOutput.read(1), std::vector<int32_t>{57});

    firstExecution.run();
    EXPECT_EQ(firstOutput.read(1), std::vector<int32_t>{23});
    EXPECT_EQ(secondOutput.read(1), std::vector<int32_t>{57});
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
    const auto expected = addVectors(firstInput, secondInput);

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
    recordAndSubmitCommands(device, queue, queueFamilyIndex,
                            [&execution](vk::CommandBuffer commandBuffer) { execution.record(commandBuffer); });

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
    const auto expected = addVectors(firstInput, secondInput);

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
    const auto bytes = makeMaxpool16x16To8x8Vgf();
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

TEST_F(VgfSessionExecutionTest, SelectsTensorTilingPerSession) {
    const auto bytes = makeMaxpool16x16To8x8Vgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const auto input = workload.resource(0);

    const Session first(context, workload);
    Session second(context, workload);
    EXPECT_EQ(first.tensorTiling(input), vk::TensorTilingARM::eLinear);

    second.setTensorTiling(input, vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(second.tensorTiling(input), vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(first.tensorTiling(input), vk::TensorTilingARM::eLinear);
}

TEST_F(VgfSessionExecutionTest, KeepsSharedSessionTilingSelectionsIndependent) {
    const auto bytes = makeOutputTensorAliasedToIntermediateTensorVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const auto input = workload.resource(0);
    const auto output = workload.resource(1);

    CompiledExecution compiled(context, workload);
    Session first(context, workload, compiled);
    Session second(context, workload, compiled);
    first.setTensorTiling(input, vk::TensorTilingARM::eOptimal);
    first.setTensorTiling(output, vk::TensorTilingARM::eOptimal);

    EXPECT_EQ(first.tensorTiling(input), vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(first.tensorTiling(output), vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(second.tensorTiling(input), vk::TensorTilingARM::eLinear);
    EXPECT_EQ(second.tensorTiling(output), vk::TensorTilingARM::eLinear);
    const ResourceCreationRecorder recorder(device);
    first.configure();
    EXPECT_EQ(recorder.pipelineTensorTilings,
              (std::vector<vk::TensorTilingARM>{vk::TensorTilingARM::eOptimal, vk::TensorTilingARM::eOptimal}));

    EXPECT_THROW(second.configure(), std::runtime_error);
    second.setTensorTiling(input, vk::TensorTilingARM::eOptimal);
    second.setTensorTiling(output, vk::TensorTilingARM::eOptimal);
    EXPECT_NO_THROW(second.configure());

    Session later(context, workload, compiled);
    EXPECT_EQ(later.tensorTiling(input), vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(later.tensorTiling(output), vk::TensorTilingARM::eOptimal);
    EXPECT_NO_THROW(later.configure());
}

TEST_F(VgfSessionExecutionTest, KeepsInternalAliasSelectionSeparateAcrossCompiledExecutions) {
    const auto bytes = makeOutputTensorAliasedToIntermediateTensorVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    Session first(context, workload);
    Session second(context, workload);
    first.setTensorTiling(workload.resource(1), vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(second.tensorTiling(workload.resource(1)), vk::TensorTilingARM::eLinear);

    const ResourceCreationRecorder recorder(device);
    first.configure();
    second.configure();
    EXPECT_EQ(recorder.pipelineTensorTilings,
              (std::vector<vk::TensorTilingARM>{vk::TensorTilingARM::eLinear, vk::TensorTilingARM::eOptimal,
                                                vk::TensorTilingARM::eLinear, vk::TensorTilingARM::eLinear}));
}

TEST_F(VgfSessionExecutionTest, RejectsConflictingPublicAliasTilings) {
    const auto bytes = makeOutputTensorAliasedToIntermediateTensorVgf(false, true);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    Session session(context, workload);
    session.setTensorTiling(workload.resource(1), vk::TensorTilingARM::eOptimal);
    session.setTensorTiling(workload.resource(2), vk::TensorTilingARM::eLinear);
    EXPECT_EQ(session.tensorTiling(workload.resource(1)), vk::TensorTilingARM::eOptimal);
    EXPECT_THROW(session.configure(), std::runtime_error);

    session.setTensorTiling(workload.resource(2), vk::TensorTilingARM::eOptimal);
    EXPECT_NO_THROW(session.configure());
}

TEST_F(VgfSessionExecutionTest, RejectsOptimalTilingWithIntermediateBufferAlias) {
    const auto bytes = makeOutputTensorAliasedToIntermediateBufferVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    Session session(context, workload);
    EXPECT_THROW(session.setTensorTiling(workload.resource(2), vk::TensorTilingARM::eOptimal), std::runtime_error);
    EXPECT_EQ(session.tensorTiling(workload.resource(2)), vk::TensorTilingARM::eLinear);
    EXPECT_NO_THROW(session.configure());
}

TEST_F(VgfSessionExecutionTest, RejectsStridedAliasPeerWithoutChangingSelectedTilings) {
    const auto bytes = makeOutputTensorAliasedToIntermediateTensorVgf(true);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    Session session(context, workload);
    EXPECT_THROW(session.setTensorTiling(workload.resource(1), vk::TensorTilingARM::eOptimal), std::runtime_error);
    EXPECT_EQ(session.tensorTiling(workload.resource(1)), vk::TensorTilingARM::eLinear);
    const ResourceCreationRecorder recorder(device);
    EXPECT_NO_THROW(session.configure());
    EXPECT_EQ(recorder.pipelineTensorTilings,
              (std::vector<vk::TensorTilingARM>{vk::TensorTilingARM::eLinear, vk::TensorTilingARM::eLinear}));
}

TEST_F(VgfSessionExecutionTest, SelectsImageTilingPerSession) {
    const auto bytes = makeAddInt32BuffersWithExternalImageVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const auto image = workload.resource(2);

    const Session first(context, workload);
    Session second(context, workload);
    EXPECT_EQ(first.imageTiling(image), vk::ImageTiling::eOptimal);

    second.setImageTiling(image, vk::ImageTiling::eLinear);
    EXPECT_EQ(second.imageTiling(image), vk::ImageTiling::eLinear);
    EXPECT_EQ(first.imageTiling(image), vk::ImageTiling::eOptimal);
}

TEST_F(VgfSessionExecutionTest, SuppliesOnlyImageLayoutMetadataForGraphImages) {
    const auto code = assembleSpirv(R"(
        OpCapability GraphARM
        OpCapability TensorsARM
        OpCapability Shader
        OpCapability VulkanMemoryModel
        OpExtension "SPV_ARM_graph"
        OpExtension "SPV_ARM_tensors"
        OpExtension "SPV_KHR_vulkan_memory_model"
        OpMemoryModel Logical Vulkan
        OpDecorate %input DescriptorSet 0
        OpDecorate %input Binding 0
        OpDecorate %output DescriptorSet 0
        OpDecorate %output Binding 1
        %float = OpTypeFloat 32
        %uint = OpTypeInt 32 0
        %zero = OpConstant %uint 0
        %one = OpConstant %uint 1
        %three = OpConstant %uint 3
        %four = OpConstant %uint 4
        %five = OpConstant %uint 5
        %dimensions = OpTypeArray %uint %four
        %shape = OpConstantComposite %dimensions %one %three %five %four
        %tensor = OpTypeTensorARM %float %four %shape
        %pointer = OpTypePointer UniformConstant %tensor
        %input = OpVariable %pointer UniformConstant
        %output = OpVariable %pointer UniformConstant
        %graphType = OpTypeGraphARM 1 %tensor %tensor
        OpGraphEntryPointARM %main "main" %input %output
        %main = OpGraphARM %graphType
        %value = OpGraphInputARM %tensor %zero
        OpGraphSetOutputARM %value %zero
        OpGraphEndARM
    )");
    auto context = wrappedContext();
    for (const bool useVgf : {false, true}) {
        SCOPED_TRACE(useVgf ? "VGF" : "standalone");
        for (const auto descriptorType :
             {vk::DescriptorType::eStorageImage, vk::DescriptorType::eCombinedImageSampler}) {
            SCOPED_TRACE(vk::to_string(descriptorType));
            const auto layout = descriptorType == vk::DescriptorType::eStorageImage
                                    ? vk::ImageLayout::eGeneral
                                    : vk::ImageLayout::eShaderReadOnlyOptimal;
            auto workload = [&] {
                if (useVgf) {
                    const auto bytes = writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
                        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "graph", "main", code);
                        const auto image = encoder.AddInputResource(static_cast<VkDescriptorType>(descriptorType),
                                                                    VK_FORMAT_R8G8B8A8_SNORM, {1, 3, 5, 4}, {});
                        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SNORM,
                                                                      {1, 3, 5, 4}, {});
                        const auto inputBinding = encoder.AddBindingSlot(0, image);
                        const auto outputBinding = encoder.AddBindingSlot(1, output);
                        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, outputBinding}, 0);
                        encoder.AddSegmentInfo(module, "image_metadata", {descriptorSet}, {inputBinding},
                                               {outputBinding}, noGraphConstants);
                    });
                    return Workload::fromVGF(bytes.data(), bytes.size());
                }
                auto imageRequirements = makeStorageImageRequirements();
                imageRequirements.descriptorType = descriptorType;
                imageRequirements.elementCount = 15;
                imageRequirements.image.extent = vk::Extent3D(5, 3, 1);
                imageRequirements.image.usage = descriptorType == vk::DescriptorType::eStorageImage
                                                    ? vk::ImageUsageFlagBits::eStorage
                                                    : vk::ImageUsageFlagBits::eSampled;
                imageRequirements.image.requiredLayout = layout;
                DataGraphDescription description;
                description.module.codeKind = ModuleCodeKind::Spirv;
                description.module.spirv = code;
                description.resources = {
                    {"image", 0, 0, ResourceAccess::Read, imageRequirements},
                    {"output", 0, 1, ResourceAccess::Write, tensorRequirements(vk::Format::eR8Snorm, {1, 3, 5, 4})},
                };
                return Workload::fromDataGraph(std::move(description));
            }();
            for (const auto tiling : {vk::ImageTiling::eLinear, vk::ImageTiling::eOptimal}) {
                SCOPED_TRACE(vk::to_string(tiling));
                Session session(context, workload);
                const auto image = workload.resource(0);
                session.setImageTiling(image, tiling);
                ResourceCreationRecorder recorder(device);
                const auto allocation = context.createImage(image, session.imageTiling(image));
                // Inspect the Vulkan arguments directly: the local graph emulation
                // layer expects tensor descriptions for every resource, including images.
                recorder.forwardPipelineCreation = false;
                EXPECT_THROW(session.configure(), std::runtime_error);

                EXPECT_EQ(recorder.pipelineResourceChains,
                          (std::vector<std::vector<vk::StructureType>>{
                              {vk::StructureType::eDataGraphPipelineResourceInfoImageLayoutARM},
                              {vk::StructureType::eTensorDescriptionARM}}));
                EXPECT_EQ(recorder.pipelineTensorTilings,
                          (std::vector<vk::TensorTilingARM>{vk::TensorTilingARM::eLinear}));
                EXPECT_EQ(recorder.pipelineTensorFormats, (std::vector<vk::Format>{vk::Format::eR8Snorm}));
                EXPECT_EQ(recorder.pipelineTensorShapes, (std::vector<std::vector<int64_t>>{{1, 3, 5, 4}}));
                EXPECT_EQ(recorder.pipelineTensorHasStrides, (std::vector<bool>{false}));
                EXPECT_EQ(recorder.pipelineImageLayouts, (std::vector<vk::ImageLayout>{layout}));
                EXPECT_EQ(recorder.imageFormats, (std::vector<vk::Format>{vk::Format::eR8G8B8A8Snorm}));
                EXPECT_EQ(recorder.imageTilings, (std::vector<vk::ImageTiling>{tiling}));
                EXPECT_EQ(image.requirements().format(), vk::Format::eR8G8B8A8Snorm);
                EXPECT_EQ(session.imageTiling(image), tiling);
            }
        }
    }
}

TEST_F(VgfSessionExecutionTest, RejectsOptimalTilingForStridedTensor) {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 1, 1});
    const auto bytes = writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "graph", "main", code);
        const auto input = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16},
                                                    {4096, 256, 16, 1});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "graph_segment", {inputSet, outputSet}, {inputBinding}, {outputBinding},
                               noGraphConstants);
    });
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const auto input = workload.resource(0);
    Session session(context, workload);

    EXPECT_THROW(session.setTensorTiling(input, vk::TensorTilingARM::eOptimal), std::runtime_error);
    EXPECT_THROW((void)context.createTensor(input, vk::TensorTilingARM::eOptimal), std::runtime_error);
    EXPECT_EQ(session.tensorTiling(input), vk::TensorTilingARM::eLinear);
}

TEST_F(VgfSessionExecutionTest, SelectsLinearTilingForAliasedImage) {
    const auto bytes = makeOutputImageAliasedToIntermediateImageVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const auto image = workload.resource(3);
    Session session(context, workload);

    const auto requirements = image.requirements().asImage();
    EXPECT_TRUE(requirements.usage(session.imageTiling(image)) & vk::ImageUsageFlagBits::eTensorAliasingARM);
    EXPECT_FALSE(requirements.usage(vk::ImageTiling::eLinear) & vk::ImageUsageFlagBits::eTensorAliasingARM);
    session.setImageTiling(image, vk::ImageTiling::eLinear);
    EXPECT_EQ(session.imageTiling(image), vk::ImageTiling::eLinear);
}

TEST_F(VgfSessionExecutionTest, RejectsGraphTensorTilingChangeAfterSharedCompilation) {
    const auto bytes = makeMaxpool16x16To8x8Vgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    CompiledExecution compiled(context, workload);

    Session first(context, workload, compiled);
    first.configure();

    // NOLINTNEXTLINE(misc-const-correctness)
    Session second(context, workload, compiled);
    EXPECT_THROW(second.setTensorTiling(workload.resource(0), vk::TensorTilingARM::eOptimal), std::runtime_error);
    EXPECT_NO_THROW(second.setTensorTiling(workload.resource(0), vk::TensorTilingARM::eLinear));
}

TEST_F(VgfSessionExecutionTest, AllowsImageTilingChangeWithSharedComputePipeline) {
    constexpr vk::DeviceSize bufferSize = 10 * sizeof(int32_t);
    const auto bytes = makeOutputImageAliasedToIntermediateImageVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();
    const auto image = workload.resource(3);
    CompiledExecution compiled(context, workload);

    Session first(context, workload, compiled);
    first.configure();
    Session second(context, workload, compiled);
    EXPECT_EQ(second.imageTiling(image), vk::ImageTiling::eOptimal);
    second.setImageTiling(image, vk::ImageTiling::eLinear);
    EXPECT_EQ(first.imageTiling(image), vk::ImageTiling::eOptimal);
    second.configure();

    const Buffer firstInput(physicalDevice, device, bufferSize);
    const Buffer secondInput(physicalDevice, device, bufferSize);
    const Buffer output(physicalDevice, device, bufferSize);
    const auto publicImage = context.createImage(image, second.imageTiling(image));
    auto bindings = second.createBindingSet();
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInput.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInput.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*output.buffer});
    auto imageBinding = publicImage.binding();
    imageBinding.layout.reset();
    bindings.bindImage(image, imageBinding);

    const ResourceCreationRecorder recorder(device);
    const auto prepared = second.prepare(bindings);
    EXPECT_EQ(recorder.imageTilings, (std::vector<vk::ImageTiling>{vk::ImageTiling::eLinear}));
}

TEST_F(VgfSessionExecutionTest, ConfigureAcceptsMissingLinearTensorAndConstantStrides) {
    const std::vector<int8_t> weights(16UL * 2 * 2 * 16, 1);
    const auto bytes = makeConv2dRescaleConstantVgf(weights);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = wrappedContext();

    ASSERT_EQ(workload.resourceCount(), 2);
    EXPECT_TRUE(workload.resource(0).requirements().asTensor().stride().empty());
    EXPECT_TRUE(workload.resource(1).requirements().asTensor().stride().empty());

    Session session(context, workload);
    EXPECT_NO_THROW(session.configure());
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
    const auto bytes = makeMaxpool16x16To8x8Vgf();
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
    const auto bytes = makeMaxpool16x16To8x8Vgf();
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

TEST_F(VgfSessionExecutionTest, RunTwoSessionsWithSharedCompiledExecution) {
    const auto bytes = makeMaxpool16x16To8x8Vgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});
    CompiledExecution compiledExecution(context, workload);

    const Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor firstOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Tensor secondInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    const Tensor secondOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});

    const auto firstInput = makeMaxpoolInput(firstInputTensor.shape, 9);
    const auto secondInput = makeMaxpoolInput(secondInputTensor.shape, 17);
    firstInputTensor.write(firstInput);
    firstOutputTensor.fill(0, firstOutputTensor.numElements());
    secondInputTensor.write(secondInput);
    secondOutputTensor.fill(0, secondOutputTensor.numElements());

    Session firstSession(context, workload, compiledExecution);
    firstSession.configure();
    auto firstBindings = firstSession.createBindingSet();
    firstBindings.bindTensor(workload.resource(0), TensorBindingInfo{*firstInputTensor.tensor});
    firstBindings.bindTensor(workload.resource(1), TensorBindingInfo{*firstOutputTensor.tensor});
    auto firstExecution = firstSession.prepare(firstBindings);

    Session secondSession(context, workload, compiledExecution);
    secondSession.configure();
    auto secondBindings = secondSession.createBindingSet();
    secondBindings.bindTensor(workload.resource(0), TensorBindingInfo{*secondInputTensor.tensor});
    secondBindings.bindTensor(workload.resource(1), TensorBindingInfo{*secondOutputTensor.tensor});
    auto secondExecution = secondSession.prepare(secondBindings);

    firstExecution.run();
    secondExecution.run();

    EXPECT_EQ(firstOutputTensor.read(firstOutputTensor.numElements()),
              expectedMaxpool(firstInput, firstInputTensor.shape));
    EXPECT_EQ(secondOutputTensor.read(secondOutputTensor.numElements()),
              expectedMaxpool(secondInput, secondInputTensor.shape));
}

TEST_F(VgfSessionExecutionTest, ExposesGraphDiagnostics) {
    const auto graphBytes = makeMaxpool16x16To8x8Vgf();
    auto graphWorkload = Workload::fromVGF(graphBytes.data(), graphBytes.size());
    auto context = wrappedContext();

    Session graphSession(context, graphWorkload);
    EXPECT_THROW(graphSession.dataGraphPipelineProperties(0), std::runtime_error);
    graphSession.configure();

    const auto graphProperties = graphSession.dataGraphPipelineProperties(0);
    ASSERT_FALSE(graphProperties.empty());
    const auto propertyData = graphSession.dataGraphPipelineProperty(0, graphProperties.front());
    EXPECT_EQ(propertyData.property, graphProperties.front());
    if (propertyData.isText) {
        ASSERT_FALSE(propertyData.data.empty());
        EXPECT_EQ(propertyData.data.back(), 0);
    }

    const auto statisticsProperty = vk::DataGraphPipelinePropertyARM::eNeuralAcceleratorStatisticsInfo;
    EXPECT_EQ(std::find(graphProperties.begin(), graphProperties.end(), statisticsProperty), graphProperties.end());
    EXPECT_THROW(graphSession.dataGraphPipelineProperty(0, statisticsProperty), std::runtime_error);
    EXPECT_THROW(graphSession.dataGraphPipelineProperties(graphWorkload.executableCount()), std::runtime_error);

    const auto sessionMemoryCount = graphSession.dataGraphSessionMemoryCount(0);
    for (uint32_t memoryIndex = 0; memoryIndex < sessionMemoryCount; ++memoryIndex) {
        const auto memoryInfo = graphSession.dataGraphSessionMemory(0, memoryIndex);
        EXPECT_NE(memoryInfo.memory, vk::DeviceMemory{});
        EXPECT_GT(memoryInfo.size, 0);
    }
    EXPECT_THROW(graphSession.dataGraphSessionMemory(0, sessionMemoryCount), std::runtime_error);

    const auto computeBytes = makeAddInt32BuffersVgf();
    auto computeWorkload = Workload::fromVGF(computeBytes.data(), computeBytes.size());
    Session computeSession(context, computeWorkload);
    computeSession.configure();
    EXPECT_THROW(computeSession.dataGraphPipelineProperties(0), std::runtime_error);
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
    std::vector<std::string> hookEvents;
    RecordOptions recordOptions;
    recordOptions.executableHook = [&hookEvents](const ExecutableRecordContext &recordContext) {
        EXPECT_NE(recordContext.commandBuffer, vk::CommandBuffer{});
        const auto *phase = recordContext.phase == ExecutableRecordPhase::Before ? "before/" : "after/";
        hookEvents.push_back(phase + std::string(recordContext.executable.name()));
    };
    recordAndSubmitCommands(device, queue, queueFamilyIndex,
                            [&execution, &recordOptions](vk::CommandBuffer commandBuffer) {
                                execution.record(commandBuffer, recordOptions);
                            });

    EXPECT_EQ(hookEvents, (std::vector<std::string>{"before/first_graph_segment", "after/first_graph_segment",
                                                    "before/second_graph_segment", "after/second_graph_segment"}));

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

TEST_P(VgfAliasExecutionTest, PrepareRejectsUndersizedAliasMemory) {
    const auto bytes = makeAliasVgf(GetParam().scenario);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    expectInvalidAliasMemory(workload, GetParam(), AliasMemoryFailure::Undersized);
}

TEST_P(VgfAliasExecutionTest, PrepareRejectsMisalignedAliasMemory) {
    const auto bytes = makeAliasVgf(GetParam().scenario);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    expectInvalidAliasMemory(workload, GetParam(), AliasMemoryFailure::Misaligned);
}

TEST_P(VgfAliasExecutionTest, PrepareRejectsOverflowingAliasMemory) {
    const auto bytes = makeAliasVgf(GetParam().scenario);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    expectInvalidAliasMemory(workload, GetParam(), AliasMemoryFailure::Overflowing);
}

INSTANTIATE_TEST_SUITE_P(AliasCases, VgfAliasExecutionTest, testing::ValuesIn(aliasCases), aliasCaseName);

#ifdef VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME
namespace {

std::string makePlaceholderGraphAndComputeVgf() {
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto graph = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool", "main");
        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, outputBinding});
        encoder.AddSegmentInfo(graph, "graph_segment", {descriptorSet}, {inputBinding}, {outputBinding},
                               noGraphConstants);

        const auto compute = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_buffers", "main");
        addInt32BuffersSegment(encoder, compute, "compute_segment");
    });
}

struct NullVulkanContext : utils::ApplicationContext {
    Context wrap() const { return Context::wrap({instance, physicalDevice, device, 0, queue}); }
};

constexpr std::array<std::byte, 17> binaryWithValidHeader = {
    std::byte{0xff}, std::byte{0xdd}, std::byte{0x22}, std::byte{0x11}, std::byte{0},   std::byte{0},
    std::byte{0},    std::byte{0},    std::byte{1},    std::byte{0},    std::byte{0},   std::byte{0},
    std::byte{0},    std::byte{0},    std::byte{0},    std::byte{0},    std::byte{0xaa}};
constexpr ArrayView<const std::byte> validBinary(binaryWithValidHeader.data(), binaryWithValidHeader.size());

} // namespace

TEST(VgfDataGraphBinary, SelectsPipelinesFromSharedBinary) {
    const auto bytes = makeTwoSegmentMaxpoolVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    Session session(context, workload);
    const std::array<uint8_t, 2> firstSelector = {17, 34};
    const std::array<uint8_t, 2> secondSelector = {51, 68};

    EXPECT_NO_THROW(session.bindDataGraphBinary(validBinary, DataGraphBinaryStorageMode::ApplicationBacked,
                                                {secondSelector.data(), secondSelector.size()}, 1));
    EXPECT_NO_THROW(session.bindDataGraphBinary(validBinary, DataGraphBinaryStorageMode::ImplementationManaged,
                                                {firstSelector.data(), firstSelector.size()}));
}

TEST(VgfDataGraphBinary, RejectsComputeAndInvalidIndices) {
    const auto bytes = makeOutputBufferAliasedToIntermediateTensorVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    Session session(context, workload);

    EXPECT_NO_THROW(session.bindDataGraphBinary(validBinary));
    EXPECT_THROW(session.bindDataGraphBinary(validBinary, DataGraphBinaryStorageMode::ImplementationManaged, {}, 1),
                 std::runtime_error);
    EXPECT_THROW(session.bindDataGraphBinary(validBinary, DataGraphBinaryStorageMode::ImplementationManaged, {}, 2),
                 std::runtime_error);
}

TEST(VgfDataGraphBinary, PreservesPlaceholderComputeBinding) {
    const auto bytes = makePlaceholderGraphAndComputeVgf();
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    ASSERT_EQ(workload.placeholderModuleCount(), 2);

    const auto bindComputeModule = [&workload](Session &session) {
        ModuleImplementation implementation;
        implementation.codeKind = ModuleCodeKind::Spirv;
        implementation.spirv = assembleAddInt32BuffersSpirv();
        session.bindModule(workload.placeholderModule(1), std::move(implementation));
    };

    Session moduleFirst(context, workload);
    bindComputeModule(moduleFirst);
    EXPECT_NO_THROW(moduleFirst.bindDataGraphBinary(validBinary));

    Session binaryFirst(context, workload);
    binaryFirst.bindDataGraphBinary(validBinary);
    EXPECT_NO_THROW(bindComputeModule(binaryFirst));
}

TEST(VgfDataGraphBinary, RejectsGraphConstantsForSelectedSegment) {
    const std::vector<int8_t> weights(16 * 2 * 2 * 16, 0);
    const auto bytes = makeConv2dRescaleConstantVgf(weights);
    auto workload = Workload::fromVGF(bytes.data(), bytes.size());
    const NullVulkanContext vulkan;
    auto context = vulkan.wrap();
    Session session(context, workload);

    EXPECT_THROW(session.bindDataGraphBinary(validBinary), std::runtime_error);
}
#endif
