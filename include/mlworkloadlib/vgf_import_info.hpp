/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace mlsdk::workloadlib {

/** @brief One model-interface binding mapped to a public workload resource. */
struct VgfInterfaceBinding {
    uint32_t binding = 0;
    uint32_t publicResourceIndex = 0;
};

/**
 * @brief Owning external-interface metadata for the workload returned by the same fromVGF() call.
 *
 * Inputs and outputs retain model-interface order, including unused entries.
 * Binding IDs are independent of segment descriptor locations and resource indices.
 * Multiple entries may refer to the same public resource; entries are not deduplicated.
 * Internal resources and constants are not exposed. Public resources absent from the
 * declared model interface have no entry; callers must not infer bindings for them.
 * The indices borrow no source memory and must only be used with that workload.
 * Successful import replaces previous contents; discard contents if import fails.
 */
struct VgfImportInfo {
    std::vector<VgfInterfaceBinding> inputs;
    std::vector<VgfInterfaceBinding> outputs;
};

} // namespace mlsdk::workloadlib
