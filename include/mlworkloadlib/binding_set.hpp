/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/array_view.hpp"
#include "mlworkloadlib/binding_types.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstddef>
#include <cstdint>
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

    /**
     * @brief Returns the borrowed original Workload associated with this binding set.
     *
     * The Workload must outlive this binding set. Throws if this binding set is invalid.
     */
    const Workload &workload() const;

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
     * @brief Places a runtime-created alias peer relative to its group's BoundMemoryInfo::offset.
     *
     * Defaults to zero. Supply the caller-resolved subresource offset when needed.
     */
    void setAliasPeerOffset(AliasPeerView peer, vk::DeviceSize byteOffset);

    /**
     * @brief Copies push-constant update bytes for an executable, replacing its previous payload.
     *
     * The executable must belong to this binding set's Workload. The payload is
     * indexed from byte offset zero; its size must be a multiple of four and
     * cannot exceed the largest compiled push-constant range end. Its pointer
     * may be null only when its size is zero.
     *
     * Recording updates only the intersections of the supplied prefix with the
     * compiled ranges, leaving other bytes untouched. Empty data explicitly
     * records no update, even when push-constant ranges exist. Omitting the
     * binding entirely still fails preparation for nonempty ranges.
     *
     * The caller must establish any other bytes read by the shader in the same
     * command buffer, following Vulkan pipeline-layout compatibility rules.
     * run() starts its own command buffer and cannot inherit caller push state.
     * The bytes are copied and prepare() snapshots them. Other executables and
     * previously prepared executions are unchanged.
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
