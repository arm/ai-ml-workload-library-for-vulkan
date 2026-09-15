/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"

#include <memory>

namespace mlsdk::workloadlib {

class Session;

/**
 * @brief Prepared binding snapshot that can run or record a workload.
 *
 * The Session and bound Vulkan objects must remain valid until submitted work
 * completes. PreparedExecution is move-only.
 */
class PreparedExecution {
  public:
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    /** @brief Releases descriptor state and runtime-created execution resources. */
    ~PreparedExecution();

    PreparedExecution(const PreparedExecution &) = delete;
    PreparedExecution &operator=(const PreparedExecution &) = delete;

    /** @brief Transfers the prepared state from another instance. */
    PreparedExecution(PreparedExecution &&) noexcept;

    /** @brief Replaces this object with prepared state transferred from another instance. */
    PreparedExecution &operator=(PreparedExecution &&) noexcept;

    /***************************************************************************
     * Execution
     **************************************************************************/

    /** @brief Records, submits, and waits for one workload execution. */
    void run();

    /**
     * @brief Records workload commands into a caller-owned command buffer.
     *
     * @p commandBuffer must be recording and compatible with the Context queue
     * family. The caller ends, submits, and synchronizes the command buffer.
     */
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

} // namespace mlsdk::workloadlib
