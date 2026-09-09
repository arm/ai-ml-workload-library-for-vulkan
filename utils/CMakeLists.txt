#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Build-tree utilities shared by this repository's samples and tests. This
# target is deliberately not installed or exported as part of the library API.
add_library(mlworkloadlib_internal_utils STATIC
    src/application_context.cpp
    src/mapped_device_memory.cpp
    src/workload_metadata.cpp
)

target_include_directories(mlworkloadlib_internal_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(mlworkloadlib_internal_utils
    PUBLIC
        mlworkloadlib
    PRIVATE
        ${CMAKE_DL_LIBS}
)

target_compile_options(mlworkloadlib_internal_utils PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})
