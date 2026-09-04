/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_types.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstddef>
#include <memory>

namespace mlworkloadlib {

class PreparedExecution;
class Session;

// Move-only resource bindings for a configured Session. Moving transfers the
// recorded bindings and Session association; the moved-from BindingSet is
// invalid. Session::prepare() snapshots the bindings. Bound Vulkan resources
// must remain alive until submitted work completes.
class BindingSet {
  public:
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    ~BindingSet();

    BindingSet(const BindingSet &) = delete;
    BindingSet &operator=(const BindingSet &) = delete;
    BindingSet(BindingSet &&) noexcept;
    BindingSet &operator=(BindingSet &&) noexcept;

    /***************************************************************************
     * Resource binding
     **************************************************************************/

    void bindTensor(ResourceView resource, TensorBindingInfo bindingInfo);
    void bindBuffer(ResourceView resource, BufferBindingInfo bindingInfo);
    void bindImage(ResourceView resource, ImageBindingInfo bindingInfo);

    void bindPushConstants(const void *data, std::size_t size);

  private:
    // Lifetime
    explicit BindingSet(Session &session);

    // Friends
    friend class Session;
    friend class PreparedExecution;

    // Implementation type
    struct Impl;

    // Implementation access
    Impl *bindingSetImpl() noexcept;
    const Impl *bindingSetImpl() const noexcept;

    // Private helpers
    const Session &session() const noexcept;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlworkloadlib
