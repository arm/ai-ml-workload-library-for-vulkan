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

namespace mlworkloadlib {

/*******************************************************************************
 * Session
 *******************************************************************************/

// Configured runtime state for a Workload on a Context.
class Session {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    Session(Context &context, const Workload &workload);

    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    /***************************************************************************
     * Configuration
     **************************************************************************/

    // Provide implementation for a workload module with missing code.
    void bindModule(PlaceholderModuleView placeholderModule, ModuleImplementation implementation);

    // Build reusable device-specific state for this session.
    void configure();

    /***************************************************************************
     * Factories
     **************************************************************************/

    // Create mutable resource bindings for this session.
    BindingSet createBindingSet();

    // Snapshot bindings and create a prepared execution.
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

} // namespace mlworkloadlib
