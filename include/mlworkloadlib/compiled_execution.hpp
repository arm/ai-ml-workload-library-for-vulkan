/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/workload.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <memory>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Session options
 *******************************************************************************/

/** @brief Pipeline compilation options. */
struct SessionOptions {
    /** @brief Borrowed pipeline cache; keep valid through the first successful Session::configure(). */
    const vk::raii::PipelineCache *pipelineCache = nullptr;
};

/*******************************************************************************
 * Compiled execution
 *******************************************************************************/

class Session;

/**
 * @brief Compiled state shared by Sessions for one Context and Workload.
 *
 * A Session builds it during configure(); later Sessions reuse it. Context and
 * Workload must outlive this object and Sessions using its state. Sessions keep
 * the compiled state alive if this object is moved or destroyed. The compiling
 * Session supplies required placeholder modules; any later bindings must
 * match. Session::configure() calls sharing this state must be serialized.
 */
class CompiledExecution final {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    /** @brief Creates empty compiled state for a Workload on one Context. */
    CompiledExecution(Context &context, const Workload &workload, SessionOptions options = {});

    /** @brief Releases this object's reference to compiled state. */
    ~CompiledExecution();

    CompiledExecution(const CompiledExecution &) = delete;
    CompiledExecution &operator=(const CompiledExecution &) = delete;

    /** @brief Moves a compiled-state handle from another object. */
    CompiledExecution(CompiledExecution &&) noexcept;

    /** @brief Replaces this object's compiled-state handle. */
    CompiledExecution &operator=(CompiledExecution &&) noexcept;

  private:
    // Implementation type
    struct Impl;

    // Friends
    friend class Session;

    // Implementation access
    std::shared_ptr<Impl> compiledExecutionImpl() const;

    // Stored state
    std::shared_ptr<Impl> impl_;
};

} // namespace mlsdk::workloadlib
