/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace mlworkloadlib::detail {

struct Module;

std::vector<uint32_t> compileGlslComputeToSpirv(const Module &module);

} // namespace mlworkloadlib::detail
