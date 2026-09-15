/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib/array_view.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace {

using mlsdk::workloadlib::ArrayView;

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST(ArrayView, ExposesMutableSpanLikeAccess) {
    std::array<int, 3> values = {1, 2, 3};

    const ArrayView<int> view(values.data(), values.size());

    EXPECT_EQ(view.data(), values.data());
    EXPECT_EQ(view.size(), values.size());
    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view.begin(), values.data());
    EXPECT_EQ(view.end(), values.data() + values.size());

    view[0] = 7;
    *view.begin() = 9;

    EXPECT_EQ(values[0], 9);
}

TEST(ArrayView, ExposesConstSpanLikeAccess) { // cppcheck-suppress syntaxError
    const std::array<int, 2> values = {4, 5};

    const ArrayView<const int> view(values.data(), values.size());

    EXPECT_EQ(view.data(), values.data());
    EXPECT_EQ(view[1], 5);
    EXPECT_EQ(*view.begin(), 4);
}

TEST(ArrayView, ExposesEmptyView) {
    const ArrayView<int> view;

    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.size(), 0);
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.begin(), view.end());
}

} // namespace
