#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(spirv-tools)
include(gtest)

add_executable(mlworkloadlib_tests
    standalone_compute/test_standalone_compute_session.cpp
    standalone_compute/test_standalone_compute_utils.hpp
    standalone_compute/test_standalone_compute_workload.cpp
    composed/test_composed_utils.hpp
    composed/test_composed_session.cpp
    standalone_data_graph/test_standalone_data_graph_session.cpp
    standalone_data_graph/test_standalone_data_graph_utils.hpp
    standalone_data_graph/test_standalone_data_graph_workload.cpp
    test_alias_execution.hpp
    test_common_utils.hpp
    test_expected_results.hpp
    test_public_utils.cpp
    test_resource_requirements.hpp
    test_spirv_utils.hpp
    test_utils.hpp
    test_vulkan_fixture.hpp
    test_vulkan_resources.hpp
    vgf/test_vgf_session.cpp
    vgf/test_vgf_utils.hpp
    vgf/test_vgf_workload.cpp
)

target_include_directories(mlworkloadlib_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(mlworkloadlib_tests PRIVATE
    GTest::gtest_main
    SPIRV-Tools-static
    VGF::vgf
    mlworkloadlib
)

target_compile_definitions(mlworkloadlib_tests PRIVATE
    ML_WORKLOAD_LIB_MAXPOOL_16X16_TO_8X8_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm/maxpool_16x16_to_8x8.spvasm"
    ML_WORKLOAD_LIB_MAXPOOL_8X8_TO_4X4_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm/maxpool_8x8_to_4x4.spvasm"
    ML_WORKLOAD_LIB_ADD_INT32_BUFFERS_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm/add_int32_buffers.spvasm"
    ML_WORKLOAD_LIB_ADD_F32_CONSTANT_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm/add_f32_constant.spvasm"
    ML_WORKLOAD_LIB_ARSHIFT_SPECBOOL_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm/arshift_specbool.spvasm"
    ML_WORKLOAD_LIB_CONV2D_RESCALE_CONSTANT_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm/conv2d_rescale_constant.spvasm"
    ML_WORKLOAD_LIB_GLSL_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources/glsl"
)

target_compile_options(mlworkloadlib_tests PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

if(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
    target_link_libraries(mlworkloadlib_tests PRIVATE mlworkloadlib_glsl_glslang)
    target_compile_definitions(mlworkloadlib_tests PRIVATE ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
endif()

if(ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
    target_link_libraries(mlworkloadlib_tests PRIVATE mlworkloadlib_hlsl_dxc)
    target_compile_definitions(mlworkloadlib_tests PRIVATE ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT)
endif()

if(NOT ANDROID)
    gtest_discover_tests(mlworkloadlib_tests PROPERTIES LABELS mlworkloadlib_tests)
endif()
