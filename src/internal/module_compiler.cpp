/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "module_compiler.hpp"
#include "workload_impl.hpp"

#include <stdexcept>

namespace mlworkloadlib::detail {
namespace {

SourceModuleCompiler hlslCompiler = nullptr;
SourceModuleCompiler glslCompiler = nullptr;

void registerGlslCompiler(SourceModuleCompiler compiler) {
    if (compiler == nullptr) {
        throw std::runtime_error("GLSL source module compiler must not be null");
    }

    glslCompiler = compiler;
}

void registerHlslCompiler(SourceModuleCompiler compiler) {
    if (compiler == nullptr) {
        throw std::runtime_error("HLSL source module compiler must not be null");
    }

    hlslCompiler = compiler;
}

} // namespace

bool supportsGlslModules() { return glslCompiler != nullptr; }

bool supportsHlslModules() { return hlslCompiler != nullptr; }

bool SourceModuleCompilerRegistration(ModuleCodeKind codeKind, SourceModuleCompiler compiler) {
    switch (codeKind) {
    case ModuleCodeKind::Glsl:
        registerGlslCompiler(compiler);
        return true;
    case ModuleCodeKind::Hlsl:
        registerHlslCompiler(compiler);
        return true;
    default:
        throw std::runtime_error("Source module compiler registration requires a source module kind");
    }
}

void compileModuleToSpirv(Module &module, ExecutableKind executableKind) {
    switch (module.codeKind) {
    case ModuleCodeKind::Spirv:
        if (module.code.empty()) {
            throw std::runtime_error("SPIR-V module code must not be empty");
        }
        return;
    case ModuleCodeKind::Glsl:
        if (executableKind != ExecutableKind::Compute) {
            throw std::runtime_error("GLSL source modules are only supported for compute executables");
        }
        if (module.source.empty()) {
            throw std::runtime_error("GLSL source module code must not be empty");
        }
        if (glslCompiler == nullptr) {
            throw std::runtime_error("GLSL source modules require a registered GLSL compiler backend");
        }
        module.code = glslCompiler(module);
        return;
    case ModuleCodeKind::Hlsl:
        if (executableKind != ExecutableKind::Compute) {
            throw std::runtime_error("HLSL source modules are only supported for compute executables");
        }
        if (module.source.empty()) {
            throw std::runtime_error("HLSL source module code must not be empty");
        }
        if (hlslCompiler == nullptr) {
            throw std::runtime_error("HLSL source modules require a registered HLSL compiler backend");
        }
        module.code = hlslCompiler(module);
        return;
    case ModuleCodeKind::Missing:
        throw std::runtime_error("Workload module requires an application-supplied implementation");
    }
    throw std::runtime_error("Unsupported workload module code kind");
}

} // namespace mlworkloadlib::detail
