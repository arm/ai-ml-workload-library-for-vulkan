/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/workload_types.hpp"

#include <string_view>
#include <vector>

namespace mlsdk::workloadlib::utils {

std::string_view resourceKindName(ResourceKind kind) noexcept;
std::string_view accessName(ResourceAccess access) noexcept;
std::string_view executableKindName(ExecutableKind kind) noexcept;

ResourceRequirements bufferRequirements(vk::DeviceSize byteSize);
ResourceRequirements tensorRequirements(vk::Format format, std::vector<int64_t> shape);

} // namespace mlsdk::workloadlib::utils
