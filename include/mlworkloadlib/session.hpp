/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"
#include "mlworkloadlib/compiled_execution.hpp"
#include "mlworkloadlib/prepared_execution.hpp"
#include "mlworkloadlib/workload.hpp"

#include <memory>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Session
 *******************************************************************************/

/**
 * @brief Configured runtime state for a Workload on a Context.
 *
 * Context and Workload are borrowed and must outlive the Session.
 */
class Session final {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    /** @brief Creates an unconfigured Session with its own compiled state. */
    Session(Context &context, const Workload &workload, SessionOptions options = {});

    /** @brief Creates an unconfigured Session sharing state with @p compiledExecution. */
    Session(Context &context, const Workload &workload, CompiledExecution &compiledExecution);

    /** @brief Releases session-specific objects and its reference to compiled state. */
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    /***************************************************************************
     * Configuration
     **************************************************************************/

    /** @brief Supplies or replaces code for a placeholder module before configure(). */
    void bindModule(PlaceholderModuleView placeholderModule, ModuleImplementation implementation);

    /** @brief Builds or reuses pipelines and layouts, then creates per-Session data graph and command objects. */
    void configure();

    /***************************************************************************
     * Factories
     **************************************************************************/

    /** @brief Creates an empty BindingSet associated with this Session. */
    BindingSet createBindingSet();

    /** @brief Validates and snapshots bindings for execution. */
    PreparedExecution prepare(const BindingSet &bindings);

  private:
    // Implementation type
    struct Impl;

    // Friends
    friend class BindingSet;
    friend class PreparedExecution;

    // Implementation access
    Impl &sessionImpl() noexcept;
    const Impl &sessionImpl() const noexcept;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlsdk::workloadlib
