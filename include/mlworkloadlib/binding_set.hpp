/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/array_view.hpp"
#include "mlworkloadlib/binding_types.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstddef>
#include <memory>
#include <vector>

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
     * @brief Copies a payload for one executable, replacing its previous payload.
     *
     * @p executable must be a valid view from this binding set's Workload.
     * The size of @p data in bytes must equal the largest offset + size of its
     * compiled push-constant ranges, or zero if there are no ranges. Its pointer
     * may be null only when its size is zero. The supplied bytes need not remain
     * valid after this call.
     * Empty data removes any stored payload for the executable and is accepted
     * only when its compiled push-constant requirement is zero.
     * Other executables' payloads and previously prepared executions are unchanged.
     */
    void bindPushConstants(ExecutableView executable, ArrayView<const std::byte> data);

    /**
     * @brief Returns a copy of an executable's bound payload, or an empty vector.
     *
     * @p executable must be a valid view from this binding set's Workload.
     * Editing the returned bytes does not change the binding.
     */
    std::vector<std::byte> pushConstants(ExecutableView executable) const;

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
    void validateExecutable(ExecutableView executable) const;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlsdk::workloadlib
