/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"
#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/prepared_execution.hpp"
#include "mlworkloadlib/workload.hpp"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <memory>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Session
 *******************************************************************************/

/**
 * @brief Configured runtime state for a Workload on a Context.
 *
 * Context and Workload are borrowed and must outlive the Session. Bind any
 * placeholder modules before calling configure().
 */
class Session final {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    /** @brief Creates an unconfigured Session for a Workload and Context. */
    Session(Context &context, const Workload &workload);

    /** @brief Releases pipelines, command state, and other device-specific session state. */
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    /***************************************************************************
     * Configuration
     **************************************************************************/

    /** @brief Supplies or replaces the implementation for a placeholder module. */
    void bindModule(PlaceholderModuleView placeholderModule, ModuleImplementation implementation);

    /** @brief Builds reusable device-specific state for every executable. */
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
