/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace mlworkloadlib::fuzz {

void fuzzWorkloadBytes(const uint8_t *data, std::size_t size);

} // namespace mlworkloadlib::fuzz
