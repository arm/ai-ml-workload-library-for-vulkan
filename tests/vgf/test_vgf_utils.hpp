/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_spirv_utils.hpp"

#include "vgf/encoder.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

namespace mlsdk::workloadlib::test {

inline const std::vector<mlsdk::vgflib::GraphConstantBindingRef> noGraphConstants;

template <typename Populate> std::string writeVgf(Populate populate) {
    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    populate(*encoder);
    encoder->Finish();

    std::stringstream stream;
    EXPECT_TRUE(encoder->WriteTo(stream));
    return stream.str();
}

inline void addInt32BuffersSegment(mlsdk::vgflib::Encoder &encoder, mlsdk::vgflib::ModuleRef module,
                                   const std::string &segmentName,
                                   const std::vector<mlsdk::vgflib::PushConstRangeRef> &pushConstantRanges = {}) {
    const auto firstInput = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
    const auto secondInput = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
    const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

    const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
    const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
    const auto outputBinding = encoder.AddBindingSlot(2, output);
    const auto inputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
    const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
    encoder.AddSegmentInfo(module, segmentName, {inputSet, outputSet}, {firstInputBinding, secondInputBinding},
                           {outputBinding}, noGraphConstants, {10, 1, 1}, pushConstantRanges);
}

inline std::string makeAddInt32BuffersVgf() {
    const auto code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);
        addInt32BuffersSegment(encoder, module, "add_int32_buffers_segment");
    });
}

inline std::string makeSingleSegmentMaxpoolVgf(const std::vector<uint32_t> &code, const std::string &moduleName,
                                               const std::vector<int64_t> &inputShape,
                                               const std::vector<int64_t> &outputShape) {
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, moduleName, "main", code);
        const auto input = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, inputShape, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, outputShape, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "maxpool_graph_segment", {inputSet, outputSet}, {inputBinding}, {outputBinding},
                               noGraphConstants);
    });
}

inline std::string makeMaxpool16x16To8x8Vgf() {
    const auto code = assembleMaxpool16x16To8x8Spirv("maxpool_16x16_to_8x8", {0, 0, 1, 1});
    return makeSingleSegmentMaxpoolVgf(code, "maxpool_16x16_to_8x8", {1, 16, 16, 16}, {1, 8, 8, 16});
}

inline std::string makeMaxpool8x8To4x4Vgf() {
    const auto code = assembleMaxpool8x8To4x4Spirv("maxpool_8x8_to_4x4", {0, 0, 1, 1});
    return makeSingleSegmentMaxpoolVgf(code, "maxpool_8x8_to_4x4", {1, 8, 8, 16}, {1, 4, 4, 16});
}

inline std::string makeTwoSegmentMaxpoolVgf() {
    const auto firstCode = assembleMaxpool16x16To8x8Spirv("first_maxpool", {0, 0, 0, 1});
    const auto secondCode = assembleMaxpool8x8To4x4Spirv("second_maxpool", {0, 0, 0, 1});
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

} // namespace mlsdk::workloadlib::test
