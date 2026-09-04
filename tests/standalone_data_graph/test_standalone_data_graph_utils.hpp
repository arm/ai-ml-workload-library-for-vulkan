/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_utils.hpp"

#include "mlworkloadlib/workload.hpp"

#include <cstdint>
#include <vector>

namespace mlworkloadlib::test {

inline DataGraphDescription makeMaxpoolDescription() {
    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleMaxpool16x16To8x8Spirv("standalone_maxpool", {0, 0, 1, 1});
    description.entryPoint = "main";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, {1, 16, 16, 16})},
        {"output", 1, 1, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16})},
    };
    return description;
}

inline std::vector<uint8_t> makeBoolSpecializationConstantData(bool value) {
    const uint32_t encodedValue = value ? 1U : 0U;
    return {static_cast<uint8_t>(encodedValue & 0xFFU), static_cast<uint8_t>((encodedValue >> 8U) & 0xFFU),
            static_cast<uint8_t>((encodedValue >> 16U) & 0xFFU), static_cast<uint8_t>((encodedValue >> 24U) & 0xFFU)};
}

inline DataGraphDescription makeArshiftSpecBoolDescription(bool round) {
    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleArshiftSpecBoolSpirv();
    description.entryPoint = "spec_arshift";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR16Uint, {4})},
        {"shift", 0, 1, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR16Uint, {4})},
        {"output", 0, 2, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR16Uint, {4})},
    };
    description.pipeline.specializationInfo.mapEntries = {vk::SpecializationMapEntry(0, 0, sizeof(uint32_t))};
    description.pipeline.specializationInfo.data = makeBoolSpecializationConstantData(round);
    return description;
}

inline DataGraphDescription makeAddF32ConstantDescription(const std::vector<float> &constant) {
    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleAddF32ConstantSpirv("standalone_add_f32_constant", {0, 0, 1, 1});
    description.entryPoint = "main";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR32Sfloat, {1, 3, 2, 1})},
        {"output", 1, 1, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR32Sfloat, {1, 3, 2, 1})},
    };
    description.constants = {{"constant", makeTensorRequirements(vk::Format::eR32Sfloat, {1, 3, 2, 1}), constant.data(),
                              constant.size() * sizeof(float)}};
    return description;
}

inline DataGraphDescription makeConv2dRescaleConstantDescription(const std::vector<int8_t> &weights) {
    DataGraphDescription description;
    description.module.codeKind = ModuleCodeKind::Spirv;
    description.module.spirv = assembleConv2dRescaleConstantSpirv("standalone_conv2d_rescale_constant", {0, 0, 1, 1});
    description.entryPoint = "main";
    description.resources = {
        {"input", 0, 0, ResourceAccess::Read, makeTensorRequirements(vk::Format::eR8Sint, {1, 16, 16, 16})},
        {"output", 1, 1, ResourceAccess::Write, makeTensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16})},
    };
    description.constants = {{"weights", makeTensorRequirements(vk::Format::eR8Sint, {16, 2, 2, 16}), weights.data(),
                              weights.size() * sizeof(int8_t)}};
    return description;
}

} // namespace mlworkloadlib::test
