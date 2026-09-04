/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"

#include <memory>

namespace mlworkloadlib {

class Session;

// Prepared binding snapshot that can run directly or record into a command buffer.
class PreparedExecution {
  public:
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    ~PreparedExecution();

    PreparedExecution(const PreparedExecution &) = delete;
    PreparedExecution &operator=(const PreparedExecution &) = delete;
    PreparedExecution(PreparedExecution &&) noexcept;
    PreparedExecution &operator=(PreparedExecution &&) noexcept;

    /***************************************************************************
     * Execution
     **************************************************************************/

    // Record, submit, and wait using runtime-owned command-buffer state.
    void run();

    // Record workload commands into a caller-owned command buffer.
    void record(vk::CommandBuffer commandBuffer);

  private:
    // Implementation type
    struct Impl;

    // Lifetime
    PreparedExecution(Session &session, const BindingSet &bindings);

    // Friends
    friend class Session;

    // Implementation access
    Impl &preparedExecutionImpl() noexcept;
    const Impl &preparedExecutionImpl() const noexcept;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlworkloadlib
