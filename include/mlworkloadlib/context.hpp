/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_types.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace mlsdk::workloadlib {

class ResourceView;
class Session;

/*******************************************************************************
 * Context metadata
 *******************************************************************************/

/**
 * @brief Application-owned Vulkan objects that can be borrowed by a Context.
 *
 * The objects must belong to the same device and remain valid while used by
 * the Context or its dependent objects.
 */
struct ContextView {
    /** @brief Constructs a view over caller-owned Vulkan objects. */
    ContextView(const vk::raii::Instance &instance, const vk::raii::PhysicalDevice &physicalDevice,
                const vk::raii::Device &device, uint32_t queueFamilyIndex, const vk::raii::Queue &queue)
        : instance(instance), physicalDevice(physicalDevice), device(device), queue(queue),
          queueFamilyIndex(queueFamilyIndex) {}

    std::reference_wrapper<const vk::raii::Instance> instance;
    std::reference_wrapper<const vk::raii::PhysicalDevice> physicalDevice;
    std::reference_wrapper<const vk::raii::Device> device;
    std::reference_wrapper<const vk::raii::Queue> queue;
    uint32_t queueFamilyIndex = 0;
};

/**
 * @brief Additional requirements for a runtime-owned Context.
 *
 * Context::create() combines these requirements with the extensions and
 * features required by the library itself. Extension names and the feature
 * chain are borrowed for the duration of the call. These requirements do not
 * apply to Context::wrap().
 */
struct RuntimeContextDeviceRequirements {
    std::vector<const char *> requiredDeviceExtensions;

    void *deviceFeaturePNext = nullptr;
};

/*******************************************************************************
 * Runtime-owned allocations
 *******************************************************************************/

/**
 * @brief Base class for a Vulkan resource and memory owned by the runtime.
 *
 * Allocations are move-only. Destroying an allocation releases its resource
 * and backing memory; therefore it must outlive any BindingSet or
 * PreparedExecution that borrows its handles.
 */
class RuntimeAllocation {
  public:
    /** @brief Returns the resource's backing-memory range. */
    BoundMemoryInfo memory() const;

    RuntimeAllocation(const RuntimeAllocation &) = delete;
    RuntimeAllocation &operator=(const RuntimeAllocation &) = delete;

    /** @brief Transfers ownership from another allocation. */
    RuntimeAllocation(RuntimeAllocation &&) noexcept;

    /** @brief Replaces this allocation with ownership transferred from another allocation. */
    RuntimeAllocation &operator=(RuntimeAllocation &&) noexcept;

  protected:
    // Implementation type
    struct Impl;

    // Lifetime
    explicit RuntimeAllocation(std::unique_ptr<Impl> impl);
    ~RuntimeAllocation();

    // Implementation access
    Impl *runtimeAllocationImpl() noexcept;
    const Impl *runtimeAllocationImpl() const noexcept;

  private:
    // Stored state
    std::unique_ptr<Impl> impl_;
};

/** @brief Runtime-owned tensor and its backing memory. */
class TensorAllocation final : public RuntimeAllocation {
  public:
    /** @brief Returns the owned tensor handle. */
    vk::TensorARM handle() const;

  private:
    // Implementation type
    struct Impl;

    // Lifetime
    TensorAllocation();

    // Friends
    friend class Context;

    // Implementation access
    Impl *tensorAllocationImpl() noexcept;
    const Impl *tensorAllocationImpl() const noexcept;
};

/** @brief Runtime-owned storage buffer and its backing memory. */
class BufferAllocation final : public RuntimeAllocation {
  public:
    /** @brief Returns the owned buffer handle. */
    vk::Buffer handle() const;

  private:
    // Implementation type
    struct Impl;

    // Lifetime
    BufferAllocation();

    // Friends
    friend class Context;

    // Implementation access
    Impl *bufferAllocationImpl() noexcept;
    const Impl *bufferAllocationImpl() const noexcept;
};

/** @brief Runtime-owned image, image view, and backing memory. */
class ImageAllocation final : public RuntimeAllocation {
  public:
    /** @brief Returns the owned image handle. */
    vk::Image handle() const;

    /**
     * @brief Returns binding information that borrows this allocation.
     *
     * The returned handles remain valid only while this allocation remains alive
     * and has not been moved from.
     */
    ImageBindingInfo binding() const;

  private:
    // Implementation type
    struct Impl;

    // Lifetime
    ImageAllocation();

    // Friends
    friend class Context;

    // Implementation access
    Impl *imageAllocationImpl() noexcept;
    const Impl *imageAllocationImpl() const noexcept;
};

/**
 * @brief Provides the Vulkan device and queue used by workload sessions.
 *
 * A Context either owns objects created by create() or borrows objects passed
 * to wrap(). It must outlive dependent Sessions and allocations.
 */
class Context {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    /** @brief Creates a Context that owns its Vulkan instance, device, and queue. */
    static Context create(const RuntimeContextDeviceRequirements &deviceRequirements = {});

    /**
     * @brief Creates a Context that borrows application-owned Vulkan objects.
     *
     * Enabled extensions and features are not validated.
     */
    static Context wrap(ContextView contextView);

    /** @brief Destroys owned Vulkan objects or releases references to wrapped objects. */
    ~Context();

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    /***************************************************************************
     * Metadata
     **************************************************************************/

    /** @brief Returns a non-owning view of this Context's Vulkan objects. */
    ContextView contextView() const;

    /***************************************************************************
     * Runtime-owned allocation
     **************************************************************************/

    /** @brief Allocates a host-visible tensor compatible with a workload resource. */
    TensorAllocation createTensor(ResourceView resource) const;

    /** @brief Allocates a host-visible buffer compatible with a workload resource. */
    BufferAllocation createBuffer(ResourceView resource) const;

    /** @brief Allocates a device-local image compatible with a workload resource. */
    ImageAllocation createImage(ResourceView resource) const;

  private:
    // Implementation type
    struct Impl;

    // Lifetime
    explicit Context(std::unique_ptr<Impl> impl);

    // Friends
    friend class Session;

    // Implementation access
    Impl &contextImpl() noexcept;
    const Impl &contextImpl() const noexcept;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlsdk::workloadlib
