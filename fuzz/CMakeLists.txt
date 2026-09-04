#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ML_WORKLOAD_LIB_ENABLE_FUZZER)
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "mlworkloadlib_fuzzer requires clang/LLVM libFuzzer.")
endif()

message(STATUS "Building mlworkloadlib_fuzzer (clang/libFuzzer + ASan + UBSan)")

add_executable(mlworkloadlib_fuzzer
    main.cpp
    workload_fuzzer.cpp
)

target_link_libraries(mlworkloadlib_fuzzer PRIVATE
    mlworkloadlib
    VGF::vgf
    VGF::vgf-utils
)

target_compile_options(mlworkloadlib PRIVATE
    -fsanitize=fuzzer-no-link,address,undefined
    -fno-sanitize-recover=undefined
    -fno-sanitize=vptr
    -fno-omit-frame-pointer
    -fno-sanitize=leak
)

target_compile_options(mlworkloadlib_fuzzer PRIVATE
    ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS}
    -fsanitize=fuzzer,address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer
    -fno-sanitize=leak
)

target_link_options(mlworkloadlib_fuzzer PRIVATE
    -fsanitize=fuzzer,address,undefined
    -fno-sanitize-recover=undefined
    -fno-sanitize=leak
)
