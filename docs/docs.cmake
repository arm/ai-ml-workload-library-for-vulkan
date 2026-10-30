#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Both the native project and docs/CMakeLists.txt use this implementation.
get_filename_component(ML_WORKLOAD_LIB_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(ML_WORKLOAD_LIB_DOCS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(ML_WORKLOAD_LIB_DOCS_DIR "${ML_WORKLOAD_LIB_DOCS_BINARY_DIR}/docs")
list(APPEND CMAKE_MODULE_PATH "${ML_WORKLOAD_LIB_REPO_ROOT}/cmake")

if(ML_WORKLOAD_LIB_DOCS_STANDALONE)
    find_package(Doxygen 1.9.1 REQUIRED)
    find_package(Sphinx REQUIRED)
else()
    find_package(Doxygen)
    find_package(Sphinx)
    if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
        message(STATUS "Doxygen or Sphinx is not available. Cannot generate documentation.")
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(WARNING "Cannot build the documentation when cross-compiling. Skipping.")
        return()
    endif()
endif()

set(SPHINX_SRC_DIR_IN "${ML_WORKLOAD_LIB_REPO_ROOT}/docs/source")
set(SPHINX_SRC_DIR "${ML_WORKLOAD_LIB_DOCS_DIR}/in")
set(SPHINX_GEN_DIR "${ML_WORKLOAD_LIB_DOCS_DIR}/generated")
set(SPHINX_SOURCES_DIR "${ML_WORKLOAD_LIB_DOCS_DIR}/sources")
set(DOC_STAGED_FILES)

# Keep the relative paths used by includes and by the combined SDK docs.
foreach(SRC_IN README.md CONTRIBUTING.md SECURITY.md LICENSES/Apache-2.0.txt)
    set(DOC_SOURCE_FILE "${SPHINX_GEN_DIR}/${SRC_IN}")
    configure_file("${ML_WORKLOAD_LIB_REPO_ROOT}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

file(GLOB_RECURSE DOC_SRC_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${SPHINX_SRC_DIR_IN}" "${SPHINX_SRC_DIR_IN}/*")
foreach(SRC_IN IN LISTS DOC_SRC_FILES)
    set(DOC_SOURCE_FILE "${SPHINX_SRC_DIR}/${SRC_IN}")
    configure_file("${SPHINX_SRC_DIR_IN}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

file(GLOB_RECURSE DOC_CODE_FILES_INCLUDE CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${ML_WORKLOAD_LIB_REPO_ROOT}" "${ML_WORKLOAD_LIB_REPO_ROOT}/include/*")
file(GLOB_RECURSE DOC_CODE_FILES_SAMPLES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${ML_WORKLOAD_LIB_REPO_ROOT}"
    "${ML_WORKLOAD_LIB_REPO_ROOT}/samples/*.cpp"
    "${ML_WORKLOAD_LIB_REPO_ROOT}/samples/*.hpp")
set(DOC_HEADER_FILES)
foreach(SRC_IN IN LISTS DOC_CODE_FILES_INCLUDE DOC_CODE_FILES_SAMPLES)
    set(DOC_SOURCE_FILE "${SPHINX_SOURCES_DIR}/${SRC_IN}")
    configure_file("${ML_WORKLOAD_LIB_REPO_ROOT}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
    if(SRC_IN MATCHES "^include/mlworkloadlib/[^/]+\\.hpp$")
        list(APPEND DOC_HEADER_FILES "${ML_WORKLOAD_LIB_REPO_ROOT}/${SRC_IN}")
    endif()
endforeach()

# Remove files that disappeared from source, including after changing revisions.
# The manifest also makes removals visible to the rendering commands.
set(DOC_STAGED_MANIFEST "${ML_WORKLOAD_LIB_DOCS_DIR}/staged-files.txt")
if(EXISTS "${DOC_STAGED_MANIFEST}")
    file(STRINGS "${DOC_STAGED_MANIFEST}" DOC_PREVIOUS_FILES)
    foreach(DOC_PREVIOUS_FILE IN LISTS DOC_PREVIOUS_FILES)
        cmake_path(IS_PREFIX ML_WORKLOAD_LIB_DOCS_DIR "${DOC_PREVIOUS_FILE}" NORMALIZE DOC_IS_STAGED)
        if(DOC_IS_STAGED AND NOT DOC_PREVIOUS_FILE IN_LIST DOC_STAGED_FILES)
            file(REMOVE "${DOC_PREVIOUS_FILE}")
        endif()
    endforeach()
endif()
list(JOIN DOC_STAGED_FILES "\n" DOC_STAGED_CONTENT)
file(CONFIGURE OUTPUT "${DOC_STAGED_MANIFEST}" CONTENT "${DOC_STAGED_CONTENT}\n" @ONLY)
# Doxygen reads the checked-in headers; the staged copies remain SDK inputs.
set(DOC_HEADER_MANIFEST "${ML_WORKLOAD_LIB_DOCS_DIR}/doxygen-inputs.txt")
list(JOIN DOC_HEADER_FILES "\n" DOC_HEADER_CONTENT)
file(CONFIGURE OUTPUT "${DOC_HEADER_MANIFEST}" CONTENT "${DOC_HEADER_CONTENT}\n" @ONLY)

include("${ML_WORKLOAD_LIB_REPO_ROOT}/cmake/doxygen.cmake")
include("${ML_WORKLOAD_LIB_REPO_ROOT}/cmake/sphinx.cmake")

add_custom_target(mlworkloadlib_doc ALL
    DEPENDS mlworkloadlib_doxy_doc mlworkloadlib_sphx_doc
    SOURCES "${SPHINX_SRC_DIR}/index.rst")
