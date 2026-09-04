#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.25)
project(MLWorkloadLibraryForVulkan
    DESCRIPTION "ML Workload Library for Vulkan"
    LANGUAGES CXX C)

include(GNUInstallDirs)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(DEFINED ANDROID_PLATFORM)
    message(WARNING "Cross-compiling for Android is currently an experimental feature.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(ML_WORKLOAD_LIB_API_VERSION_MAJOR 0)
set(ML_WORKLOAD_LIB_API_VERSION_MINOR 0)
set(ML_WORKLOAD_LIB_API_VERSION_PATCH 0)
set(ML_WORKLOAD_LIB_API_VERSION_STRING "unknown")

if(DEFINED ML_SDK_PACKAGE_VERSION AND NOT "${ML_SDK_PACKAGE_VERSION}" STREQUAL "")
    if(NOT "${ML_SDK_PACKAGE_VERSION}" MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        message(FATAL_ERROR "ML_SDK_PACKAGE_VERSION must start with major.minor.patch, got '${ML_SDK_PACKAGE_VERSION}'")
    endif()

    set(ML_WORKLOAD_LIB_API_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(ML_WORKLOAD_LIB_API_VERSION_MINOR ${CMAKE_MATCH_2})
    set(ML_WORKLOAD_LIB_API_VERSION_PATCH ${CMAKE_MATCH_3})
    set(ML_WORKLOAD_LIB_API_VERSION_STRING "${ML_SDK_PACKAGE_VERSION}")
endif()

set(ML_WORKLOAD_LIB_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/include")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/version.h.in"
    "${ML_WORKLOAD_LIB_GENERATED_INCLUDE_DIR}/mlworkloadlib/version.h"
    @ONLY)

option(ML_WORKLOAD_LIB_BUILD_TESTS "Build ML Workload Library unit tests" OFF)
option(ML_WORKLOAD_LIB_BUILD_DOCS "Build ML Workload Library documentation" OFF)
option(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT "Build ML Workload Library with GLSL source module support" OFF)
option(ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT "Build ML Workload Library with HLSL source module support" OFF)
option(ML_WORKLOAD_LIB_BUILD_SHARED "Build ML Workload Library as a shared library" OFF)
option(ML_WORKLOAD_LIB_ENABLE_FUZZER "Enable mlworkloadlib_fuzzer, it requires clang/LLVM libFuzzer." OFF)
set(ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS "${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS}" CACHE STRING
    "Additional compile options for ML Workload Library")

include(vulkan-headers)
include(vgf-lib)
if(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
    include(glslang)
endif()
if(ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
    include(dxc)
endif()

set(ML_WORKLOAD_LIB_SOURCES
    src/binding_set.cpp
    src/context.cpp
    src/internal/module_compiler.cpp
    src/internal/workload_builder.cpp
    src/prepared_execution.cpp
    src/session.cpp
    src/workload.cpp
    src/workload_formats/standalone/compute.cpp
    src/workload_formats/standalone/data_graph.cpp
    src/workload_formats/vgf.cpp
)

function(mlworkloadlib_configure_target target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${ML_WORKLOAD_LIB_GENERATED_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(${target_name} PUBLIC
        Vulkan::Headers
        VGF::vgf
        VGF::vgf-utils
    )

    target_compile_options(${target_name} PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})
endfunction()

add_library(mlworkloadlib STATIC ${ML_WORKLOAD_LIB_SOURCES})
mlworkloadlib_configure_target(mlworkloadlib)

if(ML_WORKLOAD_LIB_BUILD_SHARED)
    add_library(mlworkloadlib_shared SHARED ${ML_WORKLOAD_LIB_SOURCES})
    mlworkloadlib_configure_target(mlworkloadlib_shared)
    set_target_properties(mlworkloadlib_shared PROPERTIES OUTPUT_NAME mlworkloadlib)
endif()

if(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
    add_library(mlworkloadlib_glsl_glslang_objects OBJECT src/internal/glsl_compiler.cpp)

    target_include_directories(mlworkloadlib_glsl_glslang_objects
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(mlworkloadlib_glsl_glslang_objects
        PRIVATE
            mlworkloadlib
            "$<BUILD_LOCAL_INTERFACE:glslang::glslang>"
            "$<BUILD_LOCAL_INTERFACE:glslang::glslang-default-resource-limits>"
            "$<BUILD_LOCAL_INTERFACE:glslang::SPIRV>"
    )

    target_compile_options(mlworkloadlib_glsl_glslang_objects PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

    add_library(mlworkloadlib_glsl_glslang STATIC $<TARGET_OBJECTS:mlworkloadlib_glsl_glslang_objects>)

    target_sources(mlworkloadlib_glsl_glslang INTERFACE $<TARGET_OBJECTS:mlworkloadlib_glsl_glslang_objects>)

    target_include_directories(mlworkloadlib_glsl_glslang
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(mlworkloadlib_glsl_glslang
        PUBLIC
            mlworkloadlib
            "$<BUILD_LOCAL_INTERFACE:glslang::glslang>"
            "$<BUILD_LOCAL_INTERFACE:glslang::glslang-default-resource-limits>"
            "$<BUILD_LOCAL_INTERFACE:glslang::SPIRV>"
    )

    target_compile_options(mlworkloadlib_glsl_glslang PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})
endif()

if(ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
    add_library(mlworkloadlib_hlsl_dxc_objects OBJECT src/internal/hlsl_compiler.cpp)

    target_include_directories(mlworkloadlib_hlsl_dxc_objects
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(mlworkloadlib_hlsl_dxc_objects
        PRIVATE
            mlworkloadlib
            mlworkloadlib_dxc
    )

    target_compile_definitions(mlworkloadlib_hlsl_dxc_objects PRIVATE ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
    target_compile_options(mlworkloadlib_hlsl_dxc_objects PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

    add_library(mlworkloadlib_hlsl_dxc STATIC $<TARGET_OBJECTS:mlworkloadlib_hlsl_dxc_objects>)

    target_sources(mlworkloadlib_hlsl_dxc INTERFACE $<TARGET_OBJECTS:mlworkloadlib_hlsl_dxc_objects>)

    target_include_directories(mlworkloadlib_hlsl_dxc
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(mlworkloadlib_hlsl_dxc
        PUBLIC
            mlworkloadlib
            mlworkloadlib_dxc
    )

    target_compile_definitions(mlworkloadlib_hlsl_dxc INTERFACE ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
    target_compile_options(mlworkloadlib_hlsl_dxc PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})
endif()

add_subdirectory(samples)

install(TARGETS mlworkloadlib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
    install(TARGETS mlworkloadlib_glsl_glslang
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
    install(TARGETS mlworkloadlib_hlsl_dxc
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/binding_set.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/binding_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/prepared_execution.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/session.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/workload.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mlworkloadlib/workload_types.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlworkloadlib
)

install(FILES
    ${ML_WORKLOAD_LIB_GENERATED_INCLUDE_DIR}/mlworkloadlib/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlworkloadlib
)

if(ML_WORKLOAD_LIB_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

if(ML_SDK_BUILD_DOCS OR ML_WORKLOAD_LIB_BUILD_DOCS)
    include(docs/docs.cmake)
endif()
unset(ML_WORKLOAD_LIB_BUILD_DOCS CACHE)

if(ML_WORKLOAD_LIB_ENABLE_FUZZER)
    add_subdirectory(fuzz)
endif()
