/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "vgf-utils/temp_folder.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace mlworkloadlib::test {

/*******************************************************************************
 * Common test helpers
 *******************************************************************************/

using ::TempFolder;

inline void replaceAll(std::string &text, std::string_view from, std::string_view to) {
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

} // namespace mlworkloadlib::test
