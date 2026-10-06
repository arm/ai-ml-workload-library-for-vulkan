/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"
#include "mlworkloadlib/workload.hpp"

#include <functional>
#include <memory>

namespace mlsdk::workloadlib {

class Session;

/** @brief Point relative to an executable dispatch at which a recording hook runs. */
enum class ExecutableRecordPhase {
    Before,
    After,
};

/** @brief Context passed to a hook around one executable dispatch. */
struct ExecutableRecordContext {
    vk::CommandBuffer commandBuffer;
    ExecutableView executable;
    ExecutableRecordPhase phase;
};

/** @brief Callable invoked around one executable dispatch. */
using ExecutableRecordHook = std::function<void(const ExecutableRecordContext &)>;

/** @brief Optional caller-owned commands recorded around each executable dispatch. */
struct RecordOptions {
    ExecutableRecordHook executableHook;
};

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

    /**
     * @brief Records a workload and invokes optional hooks immediately before
     * and after each executable dispatch.
     *
     * Hooks may record commands into the supplied command buffer. Their
     * callable state and any Vulkan objects they use remain caller-owned.
     */
    void record(vk::CommandBuffer commandBuffer, const RecordOptions &options);

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
