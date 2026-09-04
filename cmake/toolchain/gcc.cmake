#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CROSSCOMPILING OFF)

find_program(GCC_PATH gcc)
if(NOT GCC_PATH)
    message(FATAL_ERROR "gcc not found")
endif()

find_program(GPP_PATH g++)
if(NOT GPP_PATH)
    message(FATAL_ERROR "g++ not found")
endif()

set(CMAKE_C_COMPILER "${GCC_PATH}" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${GPP_PATH}" CACHE FILEPATH "C++ compiler")

include(${CMAKE_CURRENT_LIST_DIR}/gnu_compiler_options.cmake)
