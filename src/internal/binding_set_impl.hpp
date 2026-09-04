/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"

#include <cstdint>
#include <map>
#include <vector>

namespace mlworkloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct BindingSet::Impl {
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    Impl(Session &sessionIn, const Workload &workloadIn) : session(sessionIn), workload(workloadIn) {}

    /***************************************************************************
     * Stored state
     **************************************************************************/

    const Session &session;
    const Workload &workload;

    std::map<uint32_t, TensorBindingInfo> tensorBindingsByResourceIndex;
    std::map<uint32_t, BufferBindingInfo> bufferBindingsByResourceIndex;
    std::map<uint32_t, ImageBindingInfo> imageBindingsByResourceIndex;

    std::vector<uint8_t> pushConstants;
};

} // namespace mlworkloadlib
