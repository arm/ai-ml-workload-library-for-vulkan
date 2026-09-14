/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_types.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstddef>
#include <memory>

namespace mlsdk::workloadlib {

class PreparedExecution;
class Session;

/**
 * @brief Move-only resource bindings for a configured Session.
 *
 * Session::prepare() snapshots the bindings. Bound Vulkan objects are borrowed
 * and must remain valid until submitted work completes.
 */
class BindingSet {
  public:
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    /** @brief Destroys the binding collection without destroying bound Vulkan objects. */
    ~BindingSet();

    BindingSet(const BindingSet &) = delete;
    BindingSet &operator=(const BindingSet &) = delete;

    /** @brief Transfers the bindings and Session association from another instance. */
    BindingSet(BindingSet &&) noexcept;

    /** @brief Replaces this instance with the bindings and Session association from another instance. */
    BindingSet &operator=(BindingSet &&) noexcept;

    /***************************************************************************
     * Resource binding
     **************************************************************************/

    /** @brief Binds a tensor, replacing any binding for the same resource. */
    void bindTensor(ResourceView resource, TensorBindingInfo bindingInfo);

    /** @brief Binds a storage buffer, replacing any binding for the same resource. */
    void bindBuffer(ResourceView resource, BufferBindingInfo bindingInfo);

    /** @brief Binds an image, replacing any binding for the same resource. */
    void bindImage(ResourceView resource, ImageBindingInfo bindingInfo);

    /**
     * @brief Copies the workload push-constant payload into this binding set.
     *
     * @p size must match the Workload requirement. @p data may be null only
     * when @p size is zero and does not need to remain valid after this call.
     */
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

} // namespace mlsdk::workloadlib
