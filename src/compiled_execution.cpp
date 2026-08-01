/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/compiled_execution_impl.hpp"

#include "mlworkloadlib/compiled_execution.hpp"

#include <memory>
#include <stdexcept>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Lifetime
 *******************************************************************************/

CompiledExecution::CompiledExecution(Context &context, const Workload &workload, SessionOptions options)
    : impl_(std::make_shared<Impl>(context, workload, options)) {}

CompiledExecution::~CompiledExecution() = default;

CompiledExecution::CompiledExecution(CompiledExecution &&) noexcept = default;

CompiledExecution &CompiledExecution::operator=(CompiledExecution &&) noexcept = default;

/*******************************************************************************
 * State access
 *******************************************************************************/

std::shared_ptr<CompiledExecution::Impl> CompiledExecution::compiledExecutionImpl() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("CompiledExecution is invalid");
    }
    return impl_;
}

} // namespace mlsdk::workloadlib
