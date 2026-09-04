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

namespace mlworkloadlib {

class ResourceView;
class Session;

/*******************************************************************************
 * Context metadata
 *******************************************************************************/

// Non-owning Vulkan object bundle used by Context::wrap().
struct ContextView {
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

struct RuntimeContextDeviceRequirements {
    std::vector<const char *> requiredDeviceExtensions;
    void *deviceFeaturePNext = nullptr;
};

/*******************************************************************************
 * Runtime-owned allocations
 *******************************************************************************/

// Runtime-owned resource allocation and backing memory.
class RuntimeAllocation {
  public:
    BoundMemoryInfo memory() const;

    RuntimeAllocation(const RuntimeAllocation &) = delete;
    RuntimeAllocation &operator=(const RuntimeAllocation &) = delete;
    RuntimeAllocation(RuntimeAllocation &&) noexcept;
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

// RAII tensor and backing memory allocated by a runtime-owned Context.
class TensorAllocation : public RuntimeAllocation {
  public:
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

// RAII buffer and backing memory allocated by a runtime-owned Context.
class BufferAllocation : public RuntimeAllocation {
  public:
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

// RAII image, view, and backing memory allocated by a runtime-owned Context.
class ImageAllocation : public RuntimeAllocation {
  public:
    vk::Image handle() const;

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

// Vulkan context used by the runtime.
class Context {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    // Create a runtime-owned Vulkan context.
    static Context create(const RuntimeContextDeviceRequirements &deviceRequirements = {});

    // Wrap caller-owned Vulkan objects without taking ownership.
    static Context wrap(ContextView contextView);

    ~Context();

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    /***************************************************************************
     * Metadata
     **************************************************************************/

    // Return borrowed Vulkan objects used by this Context.
    ContextView contextView() const;

    /***************************************************************************
     * Runtime-owned allocation
     **************************************************************************/

    // Allocate workload-compatible runtime-owned resources.
    TensorAllocation createTensor(ResourceView resource) const;
    BufferAllocation createBuffer(ResourceView resource) const;
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

} // namespace mlworkloadlib
