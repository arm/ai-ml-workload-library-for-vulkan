/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "test_common_utils.hpp"

#include <spirv-tools/libspirv.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlworkloadlib::test {

/*******************************************************************************
 * SPIR-V assembly helpers
 *******************************************************************************/

struct GraphSpvasmBindings {
    uint32_t inputSet;
    uint32_t inputBinding;
    uint32_t outputSet;
    uint32_t outputBinding;
};

inline std::vector<uint32_t> assembleSpirv(std::string_view text) {
    spvtools::SpirvTools tools{SPV_ENV_UNIVERSAL_1_6};
    if (!tools.IsValid()) {
        throw std::runtime_error("Failed to instantiate SPIR-V tools");
    }

    std::string diagnostics;
    tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &position,
                                 const char *message) {
        diagnostics += std::to_string(position.line) + ":" + std::to_string(position.column) + ": " + message + "\n";
    });

    std::vector<uint32_t> spirvModule;
    if (!tools.Assemble(std::string(text), &spirvModule)) {
        throw std::runtime_error("Failed to assemble SPIR-V program\n" + diagnostics);
    }

    if (!tools.Validate(spirvModule)) {
        throw std::runtime_error("Failed to validate SPIR-V program\n" + diagnostics);
    }

    return spirvModule;
}

inline std::vector<uint32_t> assembleGraphSpirvFromTemplate(std::string_view name, const char *templatePath,
                                                            GraphSpvasmBindings bindings) {
    std::ifstream templateFile(templatePath);
    std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    replaceAll(spvasm, "INPUT_SET", std::to_string(bindings.inputSet));
    replaceAll(spvasm, "INPUT_BINDING", std::to_string(bindings.inputBinding));
    replaceAll(spvasm, "OUTPUT_SET", std::to_string(bindings.outputSet));
    replaceAll(spvasm, "OUTPUT_BINDING", std::to_string(bindings.outputBinding));

    try {
        return assembleSpirv(spvasm);
    } catch (const std::runtime_error &error) {
        throw std::runtime_error("Failed to assemble SPIR-V test asset " + std::string(name) + ": " + error.what());
    }
}

inline std::vector<uint32_t> assembleMaxpool16x16To8x8Spirv(std::string_view name, GraphSpvasmBindings bindings) {
    return assembleGraphSpirvFromTemplate(name, ML_WORKLOAD_LIB_MAXPOOL_16X16_TO_8X8_SPVASM, bindings);
}

inline std::vector<uint32_t> assembleMaxpool8x8To4x4Spirv(std::string_view name, GraphSpvasmBindings bindings) {
    return assembleGraphSpirvFromTemplate(name, ML_WORKLOAD_LIB_MAXPOOL_8X8_TO_4X4_SPVASM, bindings);
}

inline std::vector<uint32_t> assembleAddInt32BuffersSpirv() {
    std::ifstream templateFile(ML_WORKLOAD_LIB_ADD_INT32_BUFFERS_SPVASM);
    const std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    return assembleSpirv(spvasm);
}

inline std::vector<uint32_t> assembleAddF32ConstantSpirv(std::string_view name, GraphSpvasmBindings bindings) {
    return assembleGraphSpirvFromTemplate(name, ML_WORKLOAD_LIB_ADD_F32_CONSTANT_SPVASM, bindings);
}

inline std::vector<uint32_t> assembleArshiftSpecBoolSpirv() {
    std::ifstream templateFile(ML_WORKLOAD_LIB_ARSHIFT_SPECBOOL_SPVASM);
    const std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    return assembleSpirv(spvasm);
}

inline std::vector<uint32_t> assembleConv2dRescaleConstantSpirv(std::string_view name, GraphSpvasmBindings bindings) {
    return assembleGraphSpirvFromTemplate(name, ML_WORKLOAD_LIB_CONV2D_RESCALE_CONSTANT_SPVASM, bindings);
}

} // namespace mlworkloadlib::test
