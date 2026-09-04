#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_executable(mlworkloadlib_inspect_vgf_sample
    1_inspect_vgf.cpp
)
target_link_libraries(mlworkloadlib_inspect_vgf_sample PRIVATE mlworkloadlib)
target_compile_options(mlworkloadlib_inspect_vgf_sample PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

add_executable(mlworkloadlib_data_graph_sample
    5_create_data_graph.cpp
)
target_link_libraries(mlworkloadlib_data_graph_sample PRIVATE mlworkloadlib)
target_compile_options(mlworkloadlib_data_graph_sample PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

if(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
    add_executable(mlworkloadlib_glsl_compute_sample
        2_run_glsl_compute.cpp
    )
    target_link_libraries(mlworkloadlib_glsl_compute_sample PRIVATE mlworkloadlib_glsl_glslang)
    target_compile_options(mlworkloadlib_glsl_compute_sample PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

    add_executable(mlworkloadlib_run_vgf_sample
        3_run_vgf.cpp
    )
    target_link_libraries(mlworkloadlib_run_vgf_sample PRIVATE mlworkloadlib_glsl_glslang)
    target_compile_options(mlworkloadlib_run_vgf_sample PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})

    add_executable(mlworkloadlib_wrapped_context_sample
        4_run_glsl_compute_with_wrapped_context.cpp
    )
    target_link_libraries(mlworkloadlib_wrapped_context_sample PRIVATE mlworkloadlib_glsl_glslang)
    target_compile_options(mlworkloadlib_wrapped_context_sample PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})
else()
    message(STATUS "Skipping GLSL execution samples because ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT is disabled")
endif()
