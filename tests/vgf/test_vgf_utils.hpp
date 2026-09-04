/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_utils.hpp"

#include "vgf/encoder.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace mlworkloadlib::test {

template <typename Populate> std::string writeVgf(Populate populate) {
    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    populate(*encoder);
    encoder->Finish();

    std::stringstream stream;
    EXPECT_TRUE(encoder->WriteTo(stream));
    return stream.str();
}

} // namespace mlworkloadlib::test
