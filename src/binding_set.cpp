/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/binding_set_impl.hpp"
#include "internal/session_impl.hpp"
#include "internal/utils.hpp"

#include "mlworkloadlib/binding_set.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlworkloadlib {

namespace utils = detail::utils;

namespace {

/*******************************************************************************
 * Binding validation
 *******************************************************************************/

std::string bindingError(std::string_view bindingKind, std::string_view message) {
    return std::string(bindingKind) + ": " + std::string(message);
}

void validateBindingResource(const Workload &sessionWorkload, const Workload *resourceWorkload,
                             ResourceKind resourceKind, ResourceKind expectedKind, std::string_view bindingKind) {
    if (resourceWorkload != &sessionWorkload) {
        throw std::runtime_error(bindingError(bindingKind, "workload resource belongs to a different Workload"));
    }
    if (resourceKind != expectedKind) {
        throw std::runtime_error(bindingError(
            bindingKind, "workload resource kind " + std::string(utils::resourceKindName(resourceKind)) +
                             " does not match expected " + std::string(utils::resourceKindName(expectedKind))));
    }
}

} // namespace

/*******************************************************************************
 * Lifetime
 *******************************************************************************/

BindingSet::BindingSet(Session &session) : impl_(std::make_unique<Impl>(session, session.sessionImpl().workload)) {}

BindingSet::~BindingSet() = default;

BindingSet::BindingSet(BindingSet &&) noexcept = default;

BindingSet &BindingSet::operator=(BindingSet &&) noexcept = default;

/*******************************************************************************
 * State access
 *******************************************************************************/

BindingSet::Impl *BindingSet::bindingSetImpl() noexcept { return impl_.get(); }

const BindingSet::Impl *BindingSet::bindingSetImpl() const noexcept { return impl_.get(); }

const Session &BindingSet::session() const noexcept { return impl_->session; }

/*******************************************************************************
 * Resource binding
 *******************************************************************************/

void BindingSet::bindTensor(ResourceView resource, TensorBindingInfo bindingInfo) {
    auto *bindingSetState = bindingSetImpl();
    if (bindingSetState == nullptr) {
        throw std::runtime_error(bindingError("TensorBindingInfo", "invalid BindingSet"));
    }
    validateBindingResource(bindingSetState->workload, resource.workload_, resource.requirements().kind(),
                            ResourceKind::Tensor, "TensorBindingInfo");
    bindingSetState->tensorBindingsByResourceIndex[resource.resourceIndex_] = bindingInfo;
}

void BindingSet::bindBuffer(ResourceView resource, BufferBindingInfo bindingInfo) {
    auto *bindingSetState = bindingSetImpl();
    if (bindingSetState == nullptr) {
        throw std::runtime_error(bindingError("BufferBindingInfo", "invalid BindingSet"));
    }
    validateBindingResource(bindingSetState->workload, resource.workload_, resource.requirements().kind(),
                            ResourceKind::StorageBuffer, "BufferBindingInfo");
    bindingSetState->bufferBindingsByResourceIndex[resource.resourceIndex_] = bindingInfo;
}

void BindingSet::bindImage(ResourceView resource, ImageBindingInfo bindingInfo) {
    auto *bindingSetState = bindingSetImpl();
    if (bindingSetState == nullptr) {
        throw std::runtime_error(bindingError("ImageBindingInfo", "invalid BindingSet"));
    }
    validateBindingResource(bindingSetState->workload, resource.workload_, resource.requirements().kind(),
                            ResourceKind::Image, "ImageBindingInfo");

    if (bindingInfo.imageView == nullptr) {
        throw std::runtime_error("ImageBindingInfo must provide an image view");
    }

    // Descriptor requirements
    const auto requirements = resource.requirements().asImage();
    if (requirements.requiresSamplerBinding() && bindingInfo.sampler == nullptr) {
        throw std::runtime_error("ImageBindingInfo must provide a sampler for this workload resource");
    }
    if (!requirements.requiresSamplerBinding() && bindingInfo.sampler != nullptr) {
        throw std::runtime_error("ImageBindingInfo must not provide a sampler for this workload resource");
    }
    const auto requiredSubresourceRange = requirements.requiredSubresourceRange();
    if (bindingInfo.subresourceRange.aspectMask != requiredSubresourceRange.aspectMask ||
        bindingInfo.subresourceRange.baseMipLevel != requiredSubresourceRange.baseMipLevel ||
        bindingInfo.subresourceRange.levelCount != requiredSubresourceRange.levelCount ||
        bindingInfo.subresourceRange.baseArrayLayer != requiredSubresourceRange.baseArrayLayer ||
        bindingInfo.subresourceRange.layerCount != requiredSubresourceRange.layerCount) {
        throw std::runtime_error(
            "ImageBindingInfo subresource range does not match the workload resource requirements");
    }

    bindingSetState->imageBindingsByResourceIndex[resource.resourceIndex_] = bindingInfo;
}

/*******************************************************************************
 * Push constants
 *******************************************************************************/

void BindingSet::bindPushConstants(const void *data, std::size_t size) {
    auto *bindingSetState = bindingSetImpl();
    if (bindingSetState == nullptr) {
        throw std::runtime_error(bindingError("PushConstants", "invalid BindingSet"));
    }

    if (size != 0 && data == nullptr) {
        throw std::runtime_error("Push constant payload is null");
    }

    const auto requiredSize = detail::utils::requiredPushConstantSize(bindingSetState->workload);
    if (size != requiredSize) {
        throw std::runtime_error("Push constant payload size does not match workload requirements");
    }

    if (size == 0) {
        bindingSetState->pushConstants.clear();
        return;
    }

    const auto *begin = static_cast<const uint8_t *>(data);
    bindingSetState->pushConstants.assign(begin, begin + size);
}

} // namespace mlworkloadlib
