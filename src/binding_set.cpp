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

namespace mlsdk::workloadlib {

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
            bindingKind, "workload resource kind " + std::string(detail::resourceKindName(resourceKind)) +
                             " does not match expected " + std::string(detail::resourceKindName(expectedKind))));
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

const Workload &BindingSet::workload() const {
    if (!impl_) {
        throw std::runtime_error("Cannot access the Workload of an invalid BindingSet");
    }
    return impl_->workload;
}

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

void BindingSet::setAliasPeerOffset(AliasPeerView peer, vk::DeviceSize byteOffset) {
    auto *bindingSetState = bindingSetImpl();
    if (bindingSetState == nullptr) {
        throw std::runtime_error(bindingError("AliasPeerOffset", "invalid BindingSet"));
    }
    if (peer.workload_ != &bindingSetState->workload) {
        throw std::runtime_error("AliasPeerView belongs to a different Workload");
    }
    const auto &resource = workloadImpl(bindingSetState->workload).resources.at(peer.resourceIndex_);
    if (resource.role != detail::Resource::Role::Intermediate || !resource.aliasGroupId.has_value()) {
        throw std::runtime_error("Alias peer offset requires an aliased intermediate resource");
    }
    bindingSetState->aliasPeerOffsetsByResourceIndex.insert_or_assign(peer.resourceIndex_, byteOffset);
}

/*******************************************************************************
 * Push constants
 *******************************************************************************/

void BindingSet::validateExecutable(ExecutableView executable) const {
    const auto *bindingSetState = bindingSetImpl();
    if (bindingSetState == nullptr) {
        throw std::runtime_error(bindingError("PushConstants", "invalid BindingSet"));
    }
    if (executable.workload_ != &bindingSetState->workload) {
        throw std::runtime_error(bindingError("PushConstants", "executable belongs to a different Workload"));
    }
    if (executable.index_ >= bindingSetState->session.sessionImpl().executableStates.size()) {
        throw std::runtime_error(bindingError("PushConstants", "invalid executable index"));
    }
}

void BindingSet::bindPushConstants(ExecutableView executable, ArrayView<const std::byte> data) {
    validateExecutable(executable);
    auto &bindingSetState = *bindingSetImpl();
    if (!data.empty() && data.data() == nullptr) {
        throw std::runtime_error("Push constant payload is null");
    }

    const auto &compiledExecutable =
        bindingSetState.session.sessionImpl().executableStates[executable.index_].compiledExecutable.get();
    if (data.size() != compiledExecutable.requiredPushConstantSize()) {
        throw std::runtime_error("Push constant payload size does not match executable requirements");
    }

    if (data.empty()) {
        bindingSetState.pushConstantsByExecutableIndex.erase(executable.index_);
        return;
    }

    bindingSetState.pushConstantsByExecutableIndex[executable.index_].assign(data.begin(), data.end());
}

std::vector<std::byte> BindingSet::pushConstants(ExecutableView executable) const {
    validateExecutable(executable);
    const auto &payloads = bindingSetImpl()->pushConstantsByExecutableIndex;
    const auto payload = payloads.find(executable.index_);
    return payload == payloads.end() ? std::vector<std::byte>{} : payload->second;
}

} // namespace mlsdk::workloadlib
