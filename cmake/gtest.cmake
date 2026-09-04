#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(GTEST_PATH "GTEST-NOTFOUND" CACHE PATH "Path to Google Test")

if(NOT TARGET GTest::gtest_main)
    if(EXISTS "${GTEST_PATH}/CMakeLists.txt")
        add_subdirectory("${GTEST_PATH}" googletest SYSTEM EXCLUDE_FROM_ALL)
    else()
        find_package(GTest REQUIRED)
    endif()
endif()

include(GoogleTest)
