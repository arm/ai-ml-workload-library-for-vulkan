#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(DOXYFILE_CONFIG "${ML_WORKLOAD_LIB_REPO_ROOT}/docs/Doxyfile")
set(DOXYFILE_GEN "${ML_WORKLOAD_LIB_DOCS_BINARY_DIR}/Doxyfile")
set(DOXYGEN_INPUT_DIR "${SPHINX_SOURCES_DIR}")
set(DOXYGEN_OUTPUT_DIR "${ML_WORKLOAD_LIB_DOCS_DIR}/doxygen")
set(DOXYGEN_INDEX_XML "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")
set(DOXYGEN_STAMP "${DOXYGEN_OUTPUT_DIR}/doxygen.stamp")

configure_file("${DOXYFILE_CONFIG}" "${DOXYFILE_GEN}" @ONLY)
file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}")

add_custom_command(
    OUTPUT "${DOXYGEN_STAMP}"
    BYPRODUCTS "${DOXYGEN_INDEX_XML}"
    DEPENDS "${DOXYFILE_GEN}" "${DOC_HEADER_MANIFEST}" ${DOC_HEADER_FILES}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${DOXYGEN_OUTPUT_DIR}/xml"
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYFILE_GEN}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${DOXYGEN_STAMP}"
    WORKING_DIRECTORY "${ML_WORKLOAD_LIB_DOCS_BINARY_DIR}"
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

add_custom_target(mlworkloadlib_doxy_doc DEPENDS "${DOXYGEN_STAMP}" "${DOXYGEN_INDEX_XML}")
