#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

find_package(Git)

function(mlsdk_get_git_revision SRCDIR RETURN_GIT_REVISION)
    set(${RETURN_GIT_REVISION} "unknown" PARENT_SCOPE)

    if(NOT Git_FOUND)
        message(WARNING "Git not found")
        return()
    endif()

    if(NOT IS_DIRECTORY "${SRCDIR}")
        message(WARNING "Unable to get git revision, ${SRCDIR} is not a directory")
        return()
    endif()

    # Refresh cached stat metadata before describe so clean checkouts do not
    # get a false -dirty suffix. Real local changes still report as dirty.
    execute_process(
        COMMAND ${GIT_EXECUTABLE} update-index -q --refresh
        WORKING_DIRECTORY "${SRCDIR}")

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --dirty --always --tag --broken --long
        WORKING_DIRECTORY "${SRCDIR}"
        RESULT_VARIABLE GIT_RETURN_CODE
        OUTPUT_VARIABLE GIT_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(GIT_RETURN_CODE)
        message(WARNING "Git command returns error for ${SRCDIR}")
        return()
    endif()

    if(NOT ${GIT_OUTPUT} MATCHES "^[-A-Za-z0-9/._]+$")
        message(FATAL_ERROR "Invalid revision ${GIT_OUTPUT} for ${SRCDIR}")
        return()
    endif()

    set(${RETURN_GIT_REVISION} ${GIT_OUTPUT} PARENT_SCOPE)
endfunction()
