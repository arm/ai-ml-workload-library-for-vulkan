/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/workload.hpp"

#include <cstdint>
#include <vector>

namespace mlworkloadlib::detail {

struct Module;

using SourceModuleCompiler = std::vector<uint32_t> (*)(const Module &module);

bool SourceModuleCompilerRegistration(ModuleCodeKind codeKind, SourceModuleCompiler compiler);

bool supportsGlslModules();
bool supportsHlslModules();
void compileModuleToSpirv(Module &module, ExecutableKind executableKind);

} // namespace mlworkloadlib::detail
