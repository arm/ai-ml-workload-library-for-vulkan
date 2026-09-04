/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/prepared_execution_impl.hpp"
#include "internal/utils.hpp"

#include "mlworkloadlib/prepared_execution.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlworkloadlib {

namespace utils = detail::utils;
namespace vulkan_helpers = detail::vulkan_helpers;

using DescriptorBinding = detail::DescriptorBinding;
using Resource = detail::Resource;

/*******************************************************************************
 * Runtime resource storage
 *******************************************************************************/

// Adopt memory shared by an alias group.
BoundMemoryInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptMemory(vk::raii::DeviceMemory memory,
                                                                             vk::DeviceSize size) {
    memory_.push_back(std::move(memory));
    return {*memory_.back(), 0, size};
}

// Adopt resources backed by existing memory information.
TensorBindingInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptTensor(vk::raii::TensorARM tensor,
                                                                               BoundMemoryInfo memory) {
    tensors_.push_back(std::move(tensor));
    return {*tensors_.back(), memory};
}

BufferBindingInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptBuffer(vk::raii::Buffer buffer,
                                                                               BoundMemoryInfo memory) {
    buffers_.push_back(std::move(buffer));
    return {*buffers_.back(), memory};
}

ImageBindingInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptImage(vk::raii::Image image,
                                                                             BoundMemoryInfo memory) {
    images_.push_back(std::move(image));
    return {*images_.back(), memory, {}, {}, vk::ImageLayout::eUndefined, {}};
}

// Adopt resources together with private memory owned by this storage.
TensorBindingInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptTensor(vk::raii::TensorARM tensor,
                                                                               vk::raii::DeviceMemory memory,
                                                                               vk::DeviceSize size,
                                                                               BoundMemoryInfo bindingMemory) {
    adoptMemory(std::move(memory), size);
    return adoptTensor(std::move(tensor), bindingMemory);
}

BufferBindingInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptBuffer(vk::raii::Buffer buffer,
                                                                               vk::raii::DeviceMemory memory,
                                                                               vk::DeviceSize size,
                                                                               BoundMemoryInfo bindingMemory) {
    adoptMemory(std::move(memory), size);
    return adoptBuffer(std::move(buffer), bindingMemory);
}

ImageBindingInfo PreparedExecution::Impl::RuntimeResourceStorage::adoptImage(vk::raii::Image image,
                                                                             vk::raii::DeviceMemory memory,
                                                                             vk::DeviceSize size,
                                                                             BoundMemoryInfo bindingMemory) {
    adoptMemory(std::move(memory), size);
    return adoptImage(std::move(image), bindingMemory);
}

/*******************************************************************************
 * Alias group allocation storage
 *******************************************************************************/

struct PreparedExecution::Impl::ExistingAliasGroupBindings {
    std::optional<std::reference_wrapper<const BoundTensor>> boundTensor;
    std::optional<std::reference_wrapper<const BoundBuffer>> boundBuffer;
    std::optional<std::reference_wrapper<const BoundImage>> boundImage;
    std::optional<BoundMemoryInfo> boundMemory;

    bool hasBoundAlias() const noexcept {
        return boundTensor.has_value() || boundBuffer.has_value() || boundImage.has_value();
    }
};

struct PreparedExecution::Impl::PendingAliasGroupAllocation {
    // Runtime-created alias objects
    std::vector<std::pair<DescriptorBinding, vk::raii::TensorARM>> tensors;
    std::vector<std::pair<DescriptorBinding, vk::raii::Buffer>> buffers;
    std::vector<std::pair<DescriptorBinding, vk::raii::Image>> images;

    // Shared allocation requirements
    uint32_t memoryTypeBits = std::numeric_limits<uint32_t>::max();
    vk::DeviceSize memorySize = 0;

    bool empty() const noexcept { return tensors.empty() && buffers.empty() && images.empty(); }
};

namespace {

/*******************************************************************************
 * Internal helpers
 *******************************************************************************/

void waitForFence(const vk::raii::Device &device, const vk::raii::Fence &fence) {
    const auto result = device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max());
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("vkWaitForFences failed with VkResult " +
                                 std::to_string(static_cast<int32_t>(result)));
    }
}

vk::SamplerCreateInfo makeSamplerCreateInfo(const Resource::ImageMetadata::SamplerConfig &samplerConfig) {
    return {{},
            samplerConfig.magFilter,
            samplerConfig.minFilter,
            samplerConfig.mipmapMode,
            samplerConfig.addressModeU,
            samplerConfig.addressModeV,
            samplerConfig.addressModeW,
            0.0F,
            false,
            1.0F,
            false,
            vk::CompareOp::eNever,
            0.0F,
            0.0F,
            samplerConfig.borderColor};
}

vk::ImageSubresourceRange defaultImageSubresourceRange() { return {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}; }

vk::ImageLayout requiredImageLayout(const Workload &workload, const DescriptorBinding &descBinding) {
    const auto &resource = workloadImpl(workload).resources.at(descBinding.resourceIndex);
    const auto layout = detail::imageMetadata(resource).layout;
    if (layout != vk::ImageLayout::eUndefined) {
        return layout;
    }
    return vulkan_helpers::imageLayout(utils::descriptorType(workload, descBinding));
}

template <typename RuntimeResource>
vk::MemoryRequirements resourceMemoryRequirements(const ContextView &contextView, const RuntimeResource &resource) {
    if constexpr (std::is_same_v<std::decay_t<RuntimeResource>, vk::raii::TensorARM>) {
        return contextView.device.get()
            .getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*resource))
            .memoryRequirements;
    } else {
        return resource.getMemoryRequirements();
    }
}

template <typename RuntimeResource>
void bindResourceMemory(const ContextView &contextView, RuntimeResource &resource, vk::DeviceMemory memory,
                        vk::DeviceSize offset) {
    if constexpr (std::is_same_v<std::decay_t<RuntimeResource>, vk::raii::TensorARM>) {
        contextView.device.get().bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*resource, memory, offset));
    } else {
        resource.bindMemory(memory, offset);
    }
}

vk::raii::DeviceMemory allocateDeviceLocalMemory(const ContextView &contextView,
                                                 const vk::MemoryRequirements &memoryRequirements) {
    const auto memoryType = vulkan_helpers::findMemoryType(
        contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    return {contextView.device.get(), vk::MemoryAllocateInfo(memoryRequirements.size, memoryType)};
}

/*******************************************************************************
 * Binding validation
 *******************************************************************************/

DescriptorBinding descBindingForResource(const Workload &workload, uint32_t resourceIndex) {
    const auto &resource = workloadImpl(workload).resources.at(resourceIndex);
    if (!resource.descriptorType.has_value()) {
        throw std::runtime_error("Public resource has no descriptor type");
    }
    return {resourceIndex, 0, 0, utils::resourceAccess(workload, resourceIndex)};
}

void validateBoundMemoryInfo(const Workload &workload, uint32_t resourceIndex, BoundMemoryInfo boundMemInfo) {
    const auto &resource = workloadImpl(workload).resources.at(resourceIndex);
    if ((resource.requiresBoundMemoryInfo || resource.aliasGroupId.has_value()) && boundMemInfo.memory == nullptr) {
        throw std::runtime_error("Bound memory information is required for workload resource " +
                                 std::to_string(resourceIndex));
    }
}

template <typename BoundResources>
const typename BoundResources::value_type &findBoundResource(const BoundResources &boundResources,
                                                             uint32_t resourceIndex, ResourceKind resourceKind) {
    const auto boundResource = utils::findRefIf(boundResources, [resourceIndex](const auto &resource) {
        return resource.descBinding.resourceIndex == resourceIndex;
    });
    if (!boundResource.has_value()) {
        throw std::runtime_error("No " + std::string(utils::resourceKindName(resourceKind)) +
                                 " bound for workload resource " + std::to_string(resourceIndex));
    }
    return boundResource->get();
}

/*******************************************************************************
 * Intermediate object creation
 *******************************************************************************/

vk::raii::TensorARM createIntermediateTensor(const Workload &workload, const ContextView &contextView,
                                             const DescriptorBinding &descBinding) {
    const auto &resource = workloadImpl(workload).resources.at(descBinding.resourceIndex);
    const vk::TensorDescriptionARM description(vk::TensorTilingARM::eLinear, resource.format,
                                               static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                                               resource.stride.empty() ? nullptr : resource.stride.data(),
                                               vk::TensorUsageFlagBitsARM::eDataGraph);
    const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);
    return {contextView.device.get(), createInfo};
}

vk::raii::Buffer createIntermediateBuffer(const Workload &workload, const ContextView &contextView,
                                          const DescriptorBinding &descBinding) {
    const auto &resource = workloadImpl(workload).resources.at(descBinding.resourceIndex);
    const vk::BufferCreateInfo createInfo({}, vulkan_helpers::resourceByteSize(resource),
                                          vk::BufferUsageFlagBits::eStorageBuffer);
    return {contextView.device.get(), createInfo};
}

vk::raii::Image createIntermediateImage(const Workload &workload, const ContextView &contextView,
                                        const DescriptorBinding &descBinding, bool forAliasing) {
    const auto &resource = workloadImpl(workload).resources.at(descBinding.resourceIndex);
    const auto descriptorType = utils::descriptorType(workload, descBinding);
    const vk::ImageCreateInfo createInfo(
        {}, vk::ImageType::e2D, resource.format, vulkan_helpers::imageExtentFromMetadata(resource), 1, 1,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vulkan_helpers::imageUsage(descriptorType, forAliasing),
        vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined);
    return {contextView.device.get(), createInfo};
}

void planRuntimeResourceAllocations(const Workload &workload, const std::vector<uint32_t> &executableIndices,
                                    std::vector<DescriptorBinding> &unaliasedIntermediateDescBindings,
                                    std::map<uint32_t, std::vector<DescriptorBinding>> &aliasGroups) {
    std::vector<uint32_t> plannedResourceIndices;
    const auto &workloadState = workloadImpl(workload);
    for (const auto executableIndex : executableIndices) {
        const auto &executable = workloadState.executables.at(executableIndex);
        for (const auto &descBinding : executable.bindings) {
            if (std::find(plannedResourceIndices.begin(), plannedResourceIndices.end(), descBinding.resourceIndex) !=
                plannedResourceIndices.end()) {
                continue;
            }

            const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
            if (resource.aliasGroupId.has_value()) {
                aliasGroups[*resource.aliasGroupId].push_back(descBinding);
            } else if (resource.role == Resource::Role::Intermediate) {
                unaliasedIntermediateDescBindings.push_back(descBinding);
            }
            plannedResourceIndices.push_back(descBinding.resourceIndex);
        }
    }
}

} // namespace

/*******************************************************************************
 * Construction
 *******************************************************************************/

PreparedExecution::Impl::Impl(Session &sessionIn, const BindingSet &bindingsIn) : sessionImpl(sessionIn.sessionImpl()) {
    const auto &bindingState = *bindingsIn.bindingSetImpl();

    addBindingSetResources(bindingState);
    addBoundPushConstants(bindingState);

    std::vector<uint32_t> executableIndices;
    executableIndices.reserve(sessionImpl.executableStates.size());
    for (const auto &executableState : sessionImpl.executableStates) {
        executableIndices.push_back(executableState.executableIndex);
    }

    std::vector<DescriptorBinding> unaliasedIntermediateDescBindings;
    std::map<uint32_t, std::vector<DescriptorBinding>> aliasGroups;
    planRuntimeResourceAllocations(sessionImpl.workload, executableIndices, unaliasedIntermediateDescBindings,
                                   aliasGroups);
    resolveUnaliasedIntermediateAllocations(unaliasedIntermediateDescBindings);
    resolveAliasGroups(aliasGroups);

    createDescriptorSets();
    writeDescriptors();
}

/*******************************************************************************
 * Binding setup
 *******************************************************************************/

void PreparedExecution::Impl::addBindingSetResources(const BindingSet::Impl &bindingState) {
    for (const auto &[resourceIndex, tensorBindingInfo] : bindingState.tensorBindingsByResourceIndex) {
        validateBoundMemoryInfo(sessionImpl.workload, resourceIndex, tensorBindingInfo.memory);
        addBoundTensor(tensorBindingInfo, descBindingForResource(sessionImpl.workload, resourceIndex));
    }
    for (const auto &[resourceIndex, bufferBindingInfo] : bindingState.bufferBindingsByResourceIndex) {
        validateBoundMemoryInfo(sessionImpl.workload, resourceIndex, bufferBindingInfo.memory);
        addBoundBuffer(bufferBindingInfo, descBindingForResource(sessionImpl.workload, resourceIndex));
    }
    for (const auto &[resourceIndex, imageBindingInfo] : bindingState.imageBindingsByResourceIndex) {
        validateBoundMemoryInfo(sessionImpl.workload, resourceIndex, imageBindingInfo.memory);
        addBoundImage(imageBindingInfo, descBindingForResource(sessionImpl.workload, resourceIndex));
    }
}

void PreparedExecution::Impl::addBoundPushConstants(const BindingSet::Impl &bindingState) {
    const auto requiredSize = utils::requiredPushConstantSize(sessionImpl.workload);
    const auto providedSize = bindingState.pushConstants.size();
    if (providedSize == requiredSize) {
        pushConstants = bindingState.pushConstants;
        return;
    }

    if (requiredSize == 0) {
        throw std::runtime_error("BindingSet provides push constants for a workload that does not use them");
    }
    throw std::runtime_error("BindingSet push constant size does not match workload requirements");
}

void PreparedExecution::Impl::addBoundTensor(TensorBindingInfo tensorBindingInfo, DescriptorBinding descBinding) {
    const auto &resource = workloadImpl(sessionImpl.workload).resources.at(descBinding.resourceIndex);
    const vk::TensorViewCreateInfoARM viewCreateInfo({}, tensorBindingInfo.tensor, resource.format);
    boundTensors.push_back({descBinding, tensorBindingInfo.tensor,
                            vk::raii::TensorViewARM(sessionImpl.contextView.device.get(), viewCreateInfo),
                            tensorBindingInfo.memory});
}

void PreparedExecution::Impl::addBoundBuffer(BufferBindingInfo bufferBindingInfo, DescriptorBinding descBinding) {
    boundBuffers.push_back({descBinding, bufferBindingInfo.buffer, bufferBindingInfo.memory});
}

void PreparedExecution::Impl::addBoundImage(ImageBindingInfo imageBindingInfo, DescriptorBinding descBinding) {
    const auto &resource = workloadImpl(sessionImpl.workload).resources.at(descBinding.resourceIndex);
    vulkan_helpers::validateImageFormat(resource.format);
    const auto descriptorType = utils::descriptorType(sessionImpl.workload, descBinding);

    auto subresourceRange = imageBindingInfo.subresourceRange;
    if (!subresourceRange.aspectMask) {
        subresourceRange = defaultImageSubresourceRange();
    }

    vk::raii::ImageView ownedImageView(nullptr);
    vk::ImageView imageView = imageBindingInfo.imageView;
    if (imageView == nullptr) {
        const vk::ImageViewCreateInfo viewCreateInfo({}, imageBindingInfo.image, vk::ImageViewType::e2D,
                                                     resource.format, {}, subresourceRange);
        ownedImageView = vk::raii::ImageView(sessionImpl.contextView.device.get(), viewCreateInfo);
        imageView = *ownedImageView;
    }

    vk::raii::Sampler ownedSampler(nullptr);
    vk::Sampler sampler = imageBindingInfo.sampler;
    if (descriptorType == vk::DescriptorType::eCombinedImageSampler && sampler == nullptr) {
        const auto &imageMetadata = detail::imageMetadata(resource);
        if (!imageMetadata.samplerConfig.has_value()) {
            throw std::runtime_error("Sampled image resource has no sampler config");
        }
        ownedSampler = vk::raii::Sampler(sessionImpl.contextView.device.get(),
                                         makeSamplerCreateInfo(*imageMetadata.samplerConfig));
        sampler = *ownedSampler;
    }

    boundImages.push_back({descBinding, imageBindingInfo.image, imageView, sampler, std::move(ownedImageView),
                           std::move(ownedSampler), imageBindingInfo.memory, imageBindingInfo.layout,
                           subresourceRange});
}

void PreparedExecution::Impl::resolveUnaliasedIntermediateAllocations(
    const std::vector<DescriptorBinding> &descBindings) {
    for (const auto &descBinding : descBindings) {
        const auto descriptorType = utils::descriptorType(sessionImpl.workload, descBinding);
        switch (descriptorType) {
        case vk::DescriptorType::eTensorARM:
            resolveUnaliasedIntermediateTensor(descBinding);
            break;
        case vk::DescriptorType::eStorageBuffer:
            resolveUnaliasedIntermediateBuffer(descBinding);
            break;
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eStorageImage:
            resolveUnaliasedIntermediateImage(descBinding);
            break;
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(descriptorType)));
        }
    }
}

void PreparedExecution::Impl::resolveAliasGroups(
    const std::map<uint32_t, std::vector<DescriptorBinding>> &aliasGroups) {
    for (const auto &aliasGroup : aliasGroups) {
        const auto existingBindings = findExistingAliasGroupBindings(aliasGroup.first);

        PendingAliasGroupAllocation pendingAllocation;
        for (const auto &descBinding : aliasGroup.second) {
            resolveAliasGroupBinding(descBinding, existingBindings, pendingAllocation);
        }

        if (!pendingAllocation.empty()) {
            auto memory = allocateAliasGroupMemory(pendingAllocation);
            bindAliasGroupMemory(pendingAllocation, *memory);
            const auto aliasMemory = runtimeResources.adoptMemory(std::move(memory), pendingAllocation.memorySize);
            adoptAliasGroupResources(pendingAllocation, aliasMemory);
        }
    }
}

// Resolve unaliased intermediates.
void PreparedExecution::Impl::resolveUnaliasedIntermediateTensor(const DescriptorBinding &descBinding) {
    auto tensor = createIntermediateTensor(sessionImpl.workload, sessionImpl.contextView, descBinding);
    const auto memoryRequirements = resourceMemoryRequirements(sessionImpl.contextView, tensor);
    auto memory = allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
    bindResourceMemory(sessionImpl.contextView, tensor, *memory, 0);
    const auto tensorBindingInfo =
        runtimeResources.adoptTensor(std::move(tensor), std::move(memory), memoryRequirements.size);
    addBoundTensor(tensorBindingInfo, descBinding);
}

void PreparedExecution::Impl::resolveUnaliasedIntermediateBuffer(const DescriptorBinding &descBinding) {
    auto buffer = createIntermediateBuffer(sessionImpl.workload, sessionImpl.contextView, descBinding);
    const auto memoryRequirements = resourceMemoryRequirements(sessionImpl.contextView, buffer);
    auto memory = allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
    bindResourceMemory(sessionImpl.contextView, buffer, *memory, 0);
    const auto bufferBindingInfo =
        runtimeResources.adoptBuffer(std::move(buffer), std::move(memory), memoryRequirements.size);
    addBoundBuffer(bufferBindingInfo, descBinding);
}

void PreparedExecution::Impl::resolveUnaliasedIntermediateImage(const DescriptorBinding &descBinding) {
    auto image = createIntermediateImage(sessionImpl.workload, sessionImpl.contextView, descBinding, false);
    const auto memoryRequirements = resourceMemoryRequirements(sessionImpl.contextView, image);
    auto memory = allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
    const BoundMemoryInfo bindingMemory{*memory, 0, memoryRequirements.size};
    bindResourceMemory(sessionImpl.contextView, image, bindingMemory.memory, bindingMemory.offset);
    const auto imageBindingInfo =
        runtimeResources.adoptImage(std::move(image), std::move(memory), memoryRequirements.size, bindingMemory);
    addBoundImage(imageBindingInfo, descBinding);
}

// Resolve alias groups.
PreparedExecution::Impl::ExistingAliasGroupBindings
PreparedExecution::Impl::findExistingAliasGroupBindings(uint32_t aliasGroupId) const {
    ExistingAliasGroupBindings existingBindings;
    const auto &workload = sessionImpl.workload;
    const auto findBoundResourceInAliasGroup = [&workload, aliasGroupId](auto &existingBinding,
                                                                         const auto &boundResources) {
        existingBinding = utils::findRefIf(boundResources, [&workload, aliasGroupId](const auto &boundResource) {
            const auto &resource = workloadImpl(workload).resources.at(boundResource.descBinding.resourceIndex);
            return resource.aliasGroupId.has_value() && *resource.aliasGroupId == aliasGroupId;
        });
    };

    findBoundResourceInAliasGroup(existingBindings.boundTensor, boundTensors);
    findBoundResourceInAliasGroup(existingBindings.boundBuffer, boundBuffers);
    findBoundResourceInAliasGroup(existingBindings.boundImage, boundImages);

    const auto rememberBoundMemory = [&existingBindings](const auto &existingBinding) {
        if (existingBinding.has_value() && existingBinding->get().memory.memory != nullptr) {
            existingBindings.boundMemory = existingBinding->get().memory;
        }
    };
    rememberBoundMemory(existingBindings.boundTensor);
    rememberBoundMemory(existingBindings.boundBuffer);
    rememberBoundMemory(existingBindings.boundImage);

    return existingBindings;
}

void PreparedExecution::Impl::resolveAliasGroupBinding(const DescriptorBinding &descBinding,
                                                       const ExistingAliasGroupBindings &existingBindings,
                                                       PendingAliasGroupAllocation &pendingAllocation) {
    const auto &resource = workloadImpl(sessionImpl.workload).resources.at(descBinding.resourceIndex);
    const auto descriptorType = utils::descriptorType(sessionImpl.workload, descBinding);
    if (resource.role != Resource::Role::Intermediate) {
        addExistingAliasBinding(descBinding, descriptorType, existingBindings);
        return;
    }

    if (existingBindings.hasBoundAlias() && !existingBindings.boundMemory.has_value()) {
        throw std::runtime_error("Manually bound aliases must provide memory for aliased workload resource " +
                                 std::to_string(descBinding.resourceIndex));
    }

    switch (descriptorType) {
    case vk::DescriptorType::eTensorARM: {
        auto tensor = createIntermediateTensor(sessionImpl.workload, sessionImpl.contextView, descBinding);
        if (existingBindings.boundMemory.has_value()) {
            const auto aliasMemory = *existingBindings.boundMemory;
            bindResourceMemory(sessionImpl.contextView, tensor, aliasMemory.memory, aliasMemory.offset);
            const auto tensorBindingInfo = runtimeResources.adoptTensor(std::move(tensor), aliasMemory);
            addBoundTensor(tensorBindingInfo, descBinding);
            return;
        }

        const auto requirements = resourceMemoryRequirements(sessionImpl.contextView, tensor);
        pendingAllocation.memoryTypeBits &= requirements.memoryTypeBits;
        pendingAllocation.memorySize = std::max(pendingAllocation.memorySize, requirements.size);
        pendingAllocation.tensors.emplace_back(descBinding, std::move(tensor));
        return;
    }
    case vk::DescriptorType::eStorageBuffer: {
        auto buffer = createIntermediateBuffer(sessionImpl.workload, sessionImpl.contextView, descBinding);
        if (existingBindings.boundMemory.has_value()) {
            const auto aliasMemory = *existingBindings.boundMemory;
            bindResourceMemory(sessionImpl.contextView, buffer, aliasMemory.memory, aliasMemory.offset);
            const auto bufferBindingInfo = runtimeResources.adoptBuffer(std::move(buffer), aliasMemory);
            addBoundBuffer(bufferBindingInfo, descBinding);
            return;
        }

        const auto requirements = resourceMemoryRequirements(sessionImpl.contextView, buffer);
        pendingAllocation.memoryTypeBits &= requirements.memoryTypeBits;
        pendingAllocation.memorySize = std::max(pendingAllocation.memorySize, requirements.size);
        pendingAllocation.buffers.emplace_back(descBinding, std::move(buffer));
        return;
    }
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage: {
        auto image = createIntermediateImage(sessionImpl.workload, sessionImpl.contextView, descBinding, true);
        if (existingBindings.boundMemory.has_value()) {
            const auto aliasMemory = *existingBindings.boundMemory;
            bindResourceMemory(sessionImpl.contextView, image, aliasMemory.memory, aliasMemory.offset);
            const auto imageBindingInfo = runtimeResources.adoptImage(std::move(image), aliasMemory);
            addBoundImage(imageBindingInfo, descBinding);
            return;
        }

        const auto requirements = resourceMemoryRequirements(sessionImpl.contextView, image);
        pendingAllocation.memoryTypeBits &= requirements.memoryTypeBits;
        pendingAllocation.memorySize = std::max(pendingAllocation.memorySize, requirements.size);
        pendingAllocation.images.emplace_back(descBinding, std::move(image));
        return;
    }
    default:
        throw std::runtime_error("Session does not support descriptor type " +
                                 std::to_string(static_cast<uint32_t>(descriptorType)));
    }
}

void PreparedExecution::Impl::addExistingAliasBinding(const DescriptorBinding &descBinding,
                                                      vk::DescriptorType descriptorType,
                                                      const ExistingAliasGroupBindings &existingBindings) {
    const auto throwNoAlias = [&descBinding](std::string_view resourceKind) {
        throw std::runtime_error("No manually bound " + std::string(resourceKind) + " for aliased workload resource " +
                                 std::to_string(descBinding.resourceIndex));
    };

    switch (descriptorType) {
    case vk::DescriptorType::eTensorARM: {
        if (!existingBindings.boundTensor.has_value()) {
            throwNoAlias("tensor");
        }
        const auto &boundTensor = existingBindings.boundTensor->get();
        const TensorBindingInfo tensorBindingInfo{boundTensor.tensor, boundTensor.memory};
        addBoundTensor(tensorBindingInfo, descBinding);
        return;
    }
    case vk::DescriptorType::eStorageBuffer: {
        if (!existingBindings.boundBuffer.has_value()) {
            throwNoAlias("buffer");
        }
        const auto &boundBuffer = existingBindings.boundBuffer->get();
        const BufferBindingInfo bufferBindingInfo{boundBuffer.buffer, boundBuffer.memory};
        addBoundBuffer(bufferBindingInfo, descBinding);
        return;
    }
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage: {
        if (!existingBindings.boundImage.has_value()) {
            throwNoAlias("image");
        }
        const auto &boundImage = existingBindings.boundImage->get();
        ImageBindingInfo imageBindingInfo;
        imageBindingInfo.image = boundImage.image;
        imageBindingInfo.memory = boundImage.memory;
        imageBindingInfo.imageView = boundImage.imageView;
        imageBindingInfo.sampler = boundImage.sampler;
        imageBindingInfo.layout = boundImage.layout;
        imageBindingInfo.subresourceRange = boundImage.subresourceRange;
        addBoundImage(imageBindingInfo, descBinding);
        return;
    }
    default:
        throw std::runtime_error("Session does not support descriptor type " +
                                 std::to_string(static_cast<uint32_t>(descriptorType)));
    }
}

vk::raii::DeviceMemory
PreparedExecution::Impl::allocateAliasGroupMemory(const PendingAliasGroupAllocation &pendingAllocation) const {
    vk::MemoryRequirements memoryRequirements;
    memoryRequirements.size = pendingAllocation.memorySize;
    memoryRequirements.memoryTypeBits = pendingAllocation.memoryTypeBits;

    return allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
}

void PreparedExecution::Impl::bindAliasGroupMemory(PendingAliasGroupAllocation &pendingAllocation,
                                                   vk::DeviceMemory aliasMemory) const {
    for (auto &pendingTensor : pendingAllocation.tensors) {
        bindResourceMemory(sessionImpl.contextView, pendingTensor.second, aliasMemory, 0);
    }

    for (auto &pendingBuffer : pendingAllocation.buffers) {
        bindResourceMemory(sessionImpl.contextView, pendingBuffer.second, aliasMemory, 0);
    }

    for (auto &pendingImage : pendingAllocation.images) {
        bindResourceMemory(sessionImpl.contextView, pendingImage.second, aliasMemory, 0);
    }
}

void PreparedExecution::Impl::adoptAliasGroupResources(PendingAliasGroupAllocation &pendingAllocation,
                                                       BoundMemoryInfo aliasMemory) {
    for (auto &[descBinding, tensor] : pendingAllocation.tensors) {
        const auto tensorBindingInfo = runtimeResources.adoptTensor(std::move(tensor), aliasMemory);
        addBoundTensor(tensorBindingInfo, descBinding);
    }

    for (auto &[descBinding, buffer] : pendingAllocation.buffers) {
        const auto bufferBindingInfo = runtimeResources.adoptBuffer(std::move(buffer), aliasMemory);
        addBoundBuffer(bufferBindingInfo, descBinding);
    }

    for (auto &[descBinding, image] : pendingAllocation.images) {
        const auto imageBindingInfo = runtimeResources.adoptImage(std::move(image), aliasMemory);
        addBoundImage(imageBindingInfo, descBinding);
    }
}

/*******************************************************************************
 * Descriptor setup
 *******************************************************************************/

void PreparedExecution::Impl::createDescriptorSets() {
    descriptorSetStates.reserve(sessionImpl.executableStates.size());
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    for (const auto &executableState : sessionImpl.executableStates) {
        auto &descriptors = descriptorSetStates.emplace_back();
        std::map<vk::DescriptorType, uint32_t> descriptorCounts;
        const auto &executable = workloadState.executables.at(executableState.executableIndex);
        for (const auto &descBinding : executable.bindings) {
            ++descriptorCounts[utils::descriptorType(sessionImpl.workload, descBinding)];
        }
        std::vector<vk::DescriptorPoolSize> poolSizes;
        poolSizes.reserve(descriptorCounts.size());
        for (const auto &[type, count] : descriptorCounts) {
            poolSizes.emplace_back(type, count);
        }
        descriptors.descriptorPool =
            vk::raii::DescriptorPool(sessionImpl.contextView.device.get(),
                                     {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                      static_cast<uint32_t>(executableState.descriptorSetLayouts.size()), poolSizes});
        const auto descriptorSetLayouts = vulkan_helpers::rawLayouts(executableState.descriptorSetLayouts);
        const vk::DescriptorSetAllocateInfo allocateInfo(*descriptors.descriptorPool, descriptorSetLayouts);
        descriptors.descriptorSets = sessionImpl.contextView.device.get().allocateDescriptorSets(allocateInfo);
    }
}

void PreparedExecution::Impl::writeDescriptors() const {
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    for (uint32_t executableIndex = 0; executableIndex < sessionImpl.executableStates.size(); ++executableIndex) {
        const auto &executableState = sessionImpl.executableStates[executableIndex];
        const auto &descriptorSets = descriptorSetStates[executableIndex].descriptorSets;
        const auto &executable = workloadState.executables.at(executableState.executableIndex);

        for (const auto &descBinding : executable.bindings) {
            const auto descriptorType = utils::descriptorType(sessionImpl.workload, descBinding);
            switch (descriptorType) {
            case vk::DescriptorType::eTensorARM: {
                const auto &tensor = findBoundResource(boundTensors, descBinding.resourceIndex, ResourceKind::Tensor);
                const auto tensorView = *tensor.tensorView;
                const vk::WriteDescriptorSetTensorARM tensorInfo(1, &tensorView);
                const vk::WriteDescriptorSet write(*descriptorSets[descBinding.set], descBinding.binding, 0, 1,
                                                   descriptorType, nullptr, nullptr, nullptr, &tensorInfo);
                sessionImpl.contextView.device.get().updateDescriptorSets(write, nullptr);
                break;
            }
            case vk::DescriptorType::eStorageBuffer: {
                const auto &buffer =
                    findBoundResource(boundBuffers, descBinding.resourceIndex, ResourceKind::StorageBuffer);
                const vk::DescriptorBufferInfo bufferInfo(buffer.buffer, 0, vk::WholeSize);
                const vk::WriteDescriptorSet write(*descriptorSets[descBinding.set], descBinding.binding, 0, 1,
                                                   descriptorType, nullptr, &bufferInfo);
                sessionImpl.contextView.device.get().updateDescriptorSets(write, nullptr);
                break;
            }
            case vk::DescriptorType::eCombinedImageSampler:
            case vk::DescriptorType::eStorageImage: {
                const auto &image = findBoundResource(boundImages, descBinding.resourceIndex, ResourceKind::Image);
                const vk::DescriptorImageInfo imageInfo(
                    descriptorType == vk::DescriptorType::eCombinedImageSampler ? image.sampler : vk::Sampler(),
                    image.imageView, requiredImageLayout(sessionImpl.workload, descBinding));
                const vk::WriteDescriptorSet write(*descriptorSets[descBinding.set], descBinding.binding, 0, 1,
                                                   descriptorType, &imageInfo);
                sessionImpl.contextView.device.get().updateDescriptorSets(write, nullptr);
                break;
            }
            default:
                throw std::runtime_error("Session does not support descriptor type " +
                                         std::to_string(static_cast<uint32_t>(descriptorType)));
            }
        }
    }
}

/*******************************************************************************
 * Barrier recording
 *******************************************************************************/

void PreparedExecution::Impl::insertInitialImageLayoutTransitions(vk::CommandBuffer commandBuffer) {
    // Pending transitions
    std::vector<vk::ImageMemoryBarrier2> imageBarriers;
    std::set<vk::Image> transitionedImages;
    imageBarriers.reserve(boundImages.size());
    const auto &workloadState = workloadImpl(sessionImpl.workload);

    for (auto &boundImage : boundImages) {
        // Externally managed and already transitioned images
        if (!boundImage.layout.has_value() || !transitionedImages.insert(boundImage.image).second) {
            continue;
        }

        // First workload consumer
        const auto imageResourceIndex = boundImage.descBinding.resourceIndex;
        const auto firstConsumerExecutableIt = std::find_if(
            sessionImpl.executableStates.begin(), sessionImpl.executableStates.end(),
            [&workloadState, imageResourceIndex](const auto &executableState) {
                const auto &bindings = workloadState.executables.at(executableState.executableIndex).bindings;
                return std::any_of(bindings.begin(), bindings.end(), [imageResourceIndex](const auto &descBinding) {
                    return descBinding.resourceIndex == imageResourceIndex;
                });
            });
        if (firstConsumerExecutableIt == sessionImpl.executableStates.end()) {
            throw std::runtime_error("No executable uses workload image resource " +
                                     std::to_string(imageResourceIndex));
        }

        // Image transition
        const auto oldLayout = *boundImage.layout;
        const auto descriptorLayout = requiredImageLayout(sessionImpl.workload, boundImage.descBinding);
        vk::ImageMemoryBarrier2 imageBarrier;
        imageBarrier.srcStageMask = oldLayout == vk::ImageLayout::eUndefined ? vk::PipelineStageFlagBits2::eTopOfPipe
                                                                             : vk::PipelineStageFlagBits2::eAllCommands;
        imageBarrier.srcAccessMask =
            oldLayout == vk::ImageLayout::eUndefined ? vk::AccessFlags2{} : vk::AccessFlagBits2::eMemoryWrite;
        const auto firstConsumerType = workloadState.executables.at(firstConsumerExecutableIt->executableIndex).type;
        const auto descriptorType = utils::descriptorType(sessionImpl.workload, boundImage.descBinding);
        imageBarrier.dstStageMask = vulkan_helpers::pipelineStage(firstConsumerType);
        imageBarrier.dstAccessMask = vulkan_helpers::imageAccess(firstConsumerType, descriptorType);
        imageBarrier.oldLayout = oldLayout;
        imageBarrier.newLayout = descriptorLayout;
        imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.image = boundImage.image;
        imageBarrier.subresourceRange = boundImage.subresourceRange;
        imageBarriers.push_back(imageBarrier);
        boundImage.layout = descriptorLayout;
    }

    // Barrier recording
    if (imageBarriers.empty()) {
        return;
    }

    vk::DependencyInfo dependencyInfo;
    dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    dependencyInfo.pImageMemoryBarriers = imageBarriers.data();

    const auto *const dispatcher = sessionImpl.contextView.device.get().getDispatcher();
    if (!dispatcher->vkCmdPipelineBarrier2) {
        throw std::runtime_error("vkCmdPipelineBarrier2 is not available");
    }
    dispatcher->vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(commandBuffer),
                                      reinterpret_cast<const VkDependencyInfo *>(&dependencyInfo));
}

void PreparedExecution::Impl::insertExecutableBarrier(vk::CommandBuffer commandBuffer,
                                                      const Session::Impl::ExecutableState &producer,
                                                      const Session::Impl::ExecutableState &consumer) const {
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    const auto &producerExecutable = workloadState.executables.at(producer.executableIndex);
    const auto &consumerExecutable = workloadState.executables.at(consumer.executableIndex);
    const auto producerType = producerExecutable.type;
    const auto consumerType = consumerExecutable.type;

    // Barrier buckets
    std::vector<vk::MemoryBarrier2> memoryBarriers;
    std::vector<vk::TensorMemoryBarrierARM> tensorBarriers;
    std::vector<vk::BufferMemoryBarrier2> bufferBarriers;
    std::vector<vk::ImageMemoryBarrier2> imageBarriers;
    std::vector<uint32_t> barrierAliasGroupIds;
    std::vector<uint32_t> barrierResourceIndices;

    // Producer-visible writes
    for (const auto &producerBinding : producerExecutable.bindings) {
        const auto &resource = workloadState.resources.at(producerBinding.resourceIndex);
        if ((resource.role != Resource::Role::Output && resource.role != Resource::Role::Intermediate) ||
            std::find(barrierResourceIndices.begin(), barrierResourceIndices.end(), producerBinding.resourceIndex) !=
                barrierResourceIndices.end()) {
            continue;
        }

        // Whole alias group barrier
        if (resource.aliasGroupId.has_value()) {
            if (std::find(barrierAliasGroupIds.begin(), barrierAliasGroupIds.end(), *resource.aliasGroupId) !=
                barrierAliasGroupIds.end()) {
                continue;
            }

            vk::MemoryBarrier2 memoryBarrier;
            memoryBarrier.srcStageMask = vulkan_helpers::pipelineStage(producerType);
            memoryBarrier.srcAccessMask = vulkan_helpers::writeAccess(producerType);
            memoryBarrier.dstStageMask = vulkan_helpers::pipelineStage(consumerType);
            memoryBarrier.dstAccessMask =
                vulkan_helpers::readAccess(consumerType) | vulkan_helpers::writeAccess(consumerType);

            memoryBarriers.push_back(memoryBarrier);
            barrierAliasGroupIds.push_back(*resource.aliasGroupId);
            continue;
        }

        // Resource-specific barrier
        const auto descriptorType = utils::descriptorType(sessionImpl.workload, producerBinding);
        switch (descriptorType) {
        case vk::DescriptorType::eTensorARM: {
            const auto &tensor = findBoundResource(boundTensors, producerBinding.resourceIndex, ResourceKind::Tensor);
            vk::TensorMemoryBarrierARM tensorBarrier;
            tensorBarrier.srcStageMask = vulkan_helpers::pipelineStage(producerType);
            tensorBarrier.srcAccessMask = vulkan_helpers::writeAccess(producerType);
            tensorBarrier.dstStageMask = vulkan_helpers::pipelineStage(consumerType);
            tensorBarrier.dstAccessMask =
                vulkan_helpers::readAccess(consumerType) | vulkan_helpers::writeAccess(consumerType);
            tensorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tensorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tensorBarrier.tensor = tensor.tensor;

            tensorBarriers.push_back(tensorBarrier);
            barrierResourceIndices.push_back(producerBinding.resourceIndex);
            break;
        }
        case vk::DescriptorType::eStorageBuffer: {
            const auto &buffer =
                findBoundResource(boundBuffers, producerBinding.resourceIndex, ResourceKind::StorageBuffer);
            vk::BufferMemoryBarrier2 bufferBarrier;
            bufferBarrier.srcStageMask = vulkan_helpers::pipelineStage(producerType);
            bufferBarrier.srcAccessMask = vulkan_helpers::writeAccess(producerType);
            bufferBarrier.dstStageMask = vulkan_helpers::pipelineStage(consumerType);
            bufferBarrier.dstAccessMask =
                vulkan_helpers::readAccess(consumerType) | vulkan_helpers::writeAccess(consumerType);
            bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarrier.buffer = buffer.buffer;
            bufferBarrier.offset = 0;
            bufferBarrier.size = vk::WholeSize;

            bufferBarriers.push_back(bufferBarrier);
            barrierResourceIndices.push_back(producerBinding.resourceIndex);
            break;
        }
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eStorageImage: {
            const auto &image = findBoundResource(boundImages, producerBinding.resourceIndex, ResourceKind::Image);
            const auto descriptorLayout = requiredImageLayout(sessionImpl.workload, producerBinding);
            vk::ImageMemoryBarrier2 imageBarrier;
            imageBarrier.srcStageMask = vulkan_helpers::pipelineStage(producerType);
            imageBarrier.srcAccessMask = vulkan_helpers::writeAccess(producerType);
            imageBarrier.dstStageMask = vulkan_helpers::pipelineStage(consumerType);
            imageBarrier.dstAccessMask =
                vulkan_helpers::readAccess(consumerType) | vulkan_helpers::writeAccess(consumerType);
            imageBarrier.oldLayout = descriptorLayout;
            imageBarrier.newLayout = descriptorLayout;
            imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.image = image.image;
            imageBarrier.subresourceRange = image.subresourceRange;

            imageBarriers.push_back(imageBarrier);
            barrierResourceIndices.push_back(producerBinding.resourceIndex);
            break;
        }
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(descriptorType)));
        }
    }

    // Barrier recording
    if (memoryBarriers.empty() && tensorBarriers.empty() && bufferBarriers.empty() && imageBarriers.empty()) {
        return;
    }

    vk::TensorDependencyInfoARM tensorDependencyInfo(static_cast<uint32_t>(tensorBarriers.size()),
                                                     tensorBarriers.data());
    vk::DependencyInfo dependencyInfo;
    dependencyInfo.memoryBarrierCount = static_cast<uint32_t>(memoryBarriers.size());
    dependencyInfo.pMemoryBarriers = memoryBarriers.data();
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = bufferBarriers.data();
    dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    dependencyInfo.pImageMemoryBarriers = imageBarriers.data();
    if (!tensorBarriers.empty()) {
        dependencyInfo.pNext = &tensorDependencyInfo;
    }

    const auto *const dispatcher = sessionImpl.contextView.device.get().getDispatcher();
    if (!dispatcher->vkCmdPipelineBarrier2) {
        throw std::runtime_error("vkCmdPipelineBarrier2 is not available");
    }
    dispatcher->vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(commandBuffer),
                                      reinterpret_cast<const VkDependencyInfo *>(&dependencyInfo));
}

/*******************************************************************************
 * Recording and submission
 *******************************************************************************/

void PreparedExecution::Impl::record(vk::CommandBuffer commandBuffer) {
    if (commandBuffer == nullptr) {
        throw std::runtime_error("PreparedExecution::record() requires a valid command buffer");
    }

    const auto &sessionState = sessionImpl;
    const auto *const dispatcher = sessionState.contextView.device.get().getDispatcher();
    if (!dispatcher->vkCmdBindDescriptorSets || !dispatcher->vkCmdBindPipeline || !dispatcher->vkCmdDispatch ||
        !dispatcher->vkCmdPushConstants) {
        throw std::runtime_error("Vulkan command recording functions are not available");
    }

    // Initial image layouts
    insertInitialImageLayoutTransitions(commandBuffer);

    // Executable dispatch
    const auto &workloadState = workloadImpl(sessionState.workload);
    for (size_t executableIndex = 0; executableIndex < sessionState.executableStates.size(); ++executableIndex) {
        const auto &executableState = sessionState.executableStates[executableIndex];
        const auto &executable = workloadState.executables.at(executableState.executableIndex);
        const auto &descriptors = descriptorSetStates[executableIndex];
        const auto pipelineBindPoint = vulkan_helpers::bindPoint(executable.type);
        for (uint32_t set = 0; set < static_cast<uint32_t>(descriptors.descriptorSets.size()); ++set) {
            auto *const descriptorSet = static_cast<VkDescriptorSet>(*descriptors.descriptorSets[set]);
            dispatcher->vkCmdBindDescriptorSets(
                static_cast<VkCommandBuffer>(commandBuffer), static_cast<VkPipelineBindPoint>(pipelineBindPoint),
                static_cast<VkPipelineLayout>(*executableState.pipelineLayout), set, 1, &descriptorSet, 0, nullptr);
        }
        dispatcher->vkCmdBindPipeline(static_cast<VkCommandBuffer>(commandBuffer),
                                      static_cast<VkPipelineBindPoint>(pipelineBindPoint),
                                      static_cast<VkPipeline>(*executableState.pipeline));
        for (const auto &range : executable.pushConstantRanges) {
            dispatcher->vkCmdPushConstants(static_cast<VkCommandBuffer>(commandBuffer),
                                           static_cast<VkPipelineLayout>(*executableState.pipelineLayout),
                                           static_cast<VkShaderStageFlags>(range.stageFlags), range.offset, range.size,
                                           pushConstants.data() + range.offset);
        }
        if (executable.type == ExecutableKind::Graph) {
            if (!dispatcher->vkCmdDispatchDataGraphARM) {
                throw std::runtime_error("vkCmdDispatchDataGraphARM is not available");
            }
            dispatcher->vkCmdDispatchDataGraphARM(
                static_cast<VkCommandBuffer>(commandBuffer),
                static_cast<VkDataGraphPipelineSessionARM>(*executableState.graphSession), nullptr);
        } else {
            dispatcher->vkCmdDispatch(static_cast<VkCommandBuffer>(commandBuffer), executable.dispatchShape[0],
                                      executable.dispatchShape[1], executable.dispatchShape[2]);
        }

        if (executable.implicitBarrier && executableIndex + 1 < sessionState.executableStates.size()) {
            insertExecutableBarrier(commandBuffer, executableState, sessionState.executableStates[executableIndex + 1]);
        }
    }
}

void PreparedExecution::Impl::run() {
    auto &sessionState = sessionImpl;
    waitForFence(sessionState.contextView.device.get(), sessionState.fence);
    sessionState.contextView.device.get().resetFences(*sessionState.fence);
    sessionState.commandBuffer.reset();

    sessionState.commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(*sessionState.commandBuffer);
    sessionState.commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *sessionState.commandBuffer);
    sessionState.contextView.queue.get().submit(submitInfo, *sessionState.fence);
    waitForFence(sessionState.contextView.device.get(), sessionState.fence);
}

/*******************************************************************************
 * Lifetime
 *******************************************************************************/

PreparedExecution::PreparedExecution(Session &session, const BindingSet &bindings)
    : impl_(std::make_unique<Impl>(session, bindings)) {}

PreparedExecution::~PreparedExecution() = default;

PreparedExecution::PreparedExecution(PreparedExecution &&) noexcept = default;

PreparedExecution &PreparedExecution::operator=(PreparedExecution &&) noexcept = default;

/*******************************************************************************
 * State access
 *******************************************************************************/

PreparedExecution::Impl &PreparedExecution::preparedExecutionImpl() noexcept { return *impl_; }

const PreparedExecution::Impl &PreparedExecution::preparedExecutionImpl() const noexcept { return *impl_; }

/*******************************************************************************
 * Operations
 *******************************************************************************/

void PreparedExecution::run() { preparedExecutionImpl().run(); }

void PreparedExecution::record(vk::CommandBuffer commandBuffer) { preparedExecutionImpl().record(commandBuffer); }

} // namespace mlworkloadlib
