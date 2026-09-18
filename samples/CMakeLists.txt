#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

function(mlworkloadlib_add_sample target_name source)
    add_executable(${target_name} ${source})
    target_link_libraries(${target_name} PRIVATE mlworkloadlib mlworkloadlib_internal_utils ${ARGN})
    target_compile_options(${target_name} PRIVATE ${ML_SDK_ML_WORKLOAD_LIB_COMPILE_OPTIONS})
endfunction()

mlworkloadlib_add_sample(mlworkloadlib_inspect_workload_metadata_sample 1_inspect_workload_metadata.cpp)

if(ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT)
    mlworkloadlib_add_sample(mlworkloadlib_glsl_compute_sample 2_run_glsl_compute.cpp
        mlworkloadlib_glsl_glslang
    )

    if(ML_WORKLOAD_LIB_ENABLE_VGF_SUPPORT)
        mlworkloadlib_add_sample(mlworkloadlib_run_vgf_with_placeholder_module_sample
            3_run_vgf_with_placeholder_module.cpp
            mlworkloadlib_glsl_glslang
            VGF::vgf
        )
    endif()

    mlworkloadlib_add_sample(mlworkloadlib_record_glsl_compute_with_wrapped_context_sample
        4_record_glsl_compute_with_wrapped_context.cpp
        mlworkloadlib_glsl_glslang
    )
else()
    message(STATUS "Skipping GLSL execution samples because ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT is disabled")
endif()

mlworkloadlib_add_sample(mlworkloadlib_create_standalone_data_graph_sample 5_create_standalone_data_graph.cpp)
