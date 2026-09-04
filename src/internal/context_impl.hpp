/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/context.hpp"

#include <limits>

namespace mlworkloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct Context::Impl {
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    Impl() = default;
    virtual ~Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    Impl(Impl &&) = delete;
    Impl &operator=(Impl &&) = delete;

    /***************************************************************************
     * Context access
     **************************************************************************/

    virtual ContextView contextView() const = 0;

    /***************************************************************************
     * Implementation variants
     **************************************************************************/

    struct Owned;
    struct Wrapped;
};

struct Context::Impl::Owned final : Context::Impl {
    /***************************************************************************
     * Context access
     **************************************************************************/

    ContextView contextView() const override { return {instance, physicalDevice, device, queueFamilyIndex, queue}; }

    /***************************************************************************
     * Stored state
     **************************************************************************/

    vk::raii::Context raiiContext;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue queue{nullptr};

    uint32_t queueFamilyIndex = std::numeric_limits<uint32_t>::max();
};

struct Context::Impl::Wrapped final : Context::Impl {
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    explicit Wrapped(ContextView contextIn) : context(contextIn) {}

    /***************************************************************************
     * Context access
     **************************************************************************/

    ContextView contextView() const override { return context; }

    /***************************************************************************
     * Stored state
     **************************************************************************/

    ContextView context;
};

} // namespace mlworkloadlib
