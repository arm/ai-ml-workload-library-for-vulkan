/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/alias_memory.hpp"
#include "internal/prepared_execution_impl.hpp"
#include "internal/runtime_resource.hpp"
#include "internal/session_tiling.hpp"
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlsdk::workloadlib {

using DescriptorBinding = detail::DescriptorBinding;
using Resource = detail::Resource;
using detail::aliasMemoryRequirements;
using detail::createIntermediateBuffer;
using detail::createIntermediateImage;
using detail::createIntermediateTensor;

struct PreparedExecution::Impl::BoundTensor {
    DescriptorBinding descBinding;
    vk::TensorARM tensor{nullptr};
    vk::raii::TensorViewARM tensorView{nullptr};
    BoundMemoryInfo memory{};
};

struct PreparedExecution::Impl::BoundBuffer {
    DescriptorBinding descBinding;
    vk::Buffer buffer{nullptr};
    BoundMemoryInfo memory{};
};

struct PreparedExecution::Impl::BoundImage {
    // Bound handles
    DescriptorBinding descBinding;
    vk::Image image{nullptr};
    vk::ImageView imageView{nullptr};
    vk::Sampler sampler{nullptr};

    // Runtime-owned descriptor helpers
    vk::raii::ImageView ownedImageView{nullptr};
    vk::raii::Sampler ownedSampler{nullptr};

    // Memory and layout metadata
    BoundMemoryInfo memory{};
    std::optional<vk::ImageLayout> layout;
    vk::ImageSubresourceRange subresourceRange;
};

struct PreparedExecution::Impl::DescriptorSetState {
    vk::raii::DescriptorPool descriptorPool{nullptr};
    std::vector<vk::raii::DescriptorSet> descriptorSets;
};

struct PreparedExecution::Impl::ResourceUse {
    uint32_t resourceIndex = 0;
    vk::AccessFlags2 accessMask;
    bool writes = false;
};

struct PreparedExecution::Impl::SynchronizationState {
    struct ResourceInfo {
        uint32_t synchronizationSlotIndex = 0;
        std::optional<uint32_t> imageStateIndex;
        bool isOptimalTensor = false;
    };

    struct Slot {
        bool isAliasGroup = false;
        std::vector<uint32_t> imageStateIndices;

        vk::PipelineStageFlags2 pendingStageMask;
        vk::AccessFlags2 pendingAccessMask;
        bool pendingWrites = false;
        bool used = false;
        bool lastImplicitBarrier = true;

        void recordUse(const ResourceUse &use, vk::PipelineStageFlags2 stageMask, bool implicitBarrier,
                       bool imageBarrierEmitted);
    };

    struct Image {
        vk::Image image{nullptr};
        vk::ImageSubresourceRange subresourceRange;
        std::optional<vk::ImageLayout> initialLayout;
        vk::ImageLayout currentLayout = vk::ImageLayout::eUndefined;
        vk::ImageLayout descriptorLayout = vk::ImageLayout::eUndefined;
        uint32_t synchronizationSlotIndex = 0;
        bool used = false;
    };

    std::vector<ResourceInfo> resources;
    std::vector<Slot> slots;
    std::vector<Image> images;

    std::vector<ResourceUse> collectResourceUses(const detail::Executable &executable) const;
    BarrierBatch buildExecutableBarriers(const detail::Executable &executable, const std::vector<ResourceUse> &uses);
    BarrierBatch buildExitBarriers();

    bool prepareImageAccess(BarrierBatch &barriers, const ResourceInfo &resource, vk::PipelineStageFlags2 stageMask,
                            vk::AccessFlags2 accessMask);
    // Returns whether a barrier was emitted, including same-layout entry barriers.
    bool ensureImageLayout(BarrierBatch &barriers, uint32_t imageIndex, vk::ImageLayout layout,
                           vk::PipelineStageFlags2 dstStageMask, vk::AccessFlags2 dstAccessMask);
};

struct PreparedExecution::Impl::BarrierBatch {
    std::vector<vk::MemoryBarrier2> memoryBarriers;
    std::vector<vk::ImageMemoryBarrier2> imageBarriers;

    void record(const vk::raii::Device &device, vk::CommandBuffer commandBuffer) const;
};

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
    // Snapshot handles because adding peers can reallocate the bound-resource vectors.
    std::optional<TensorBindingInfo> boundTensor;
    std::optional<BufferBindingInfo> boundBuffer;
    std::optional<ImageBindingInfo> boundImage;
    std::optional<BoundMemoryInfo> boundMemory;
    uint32_t memoryTypeBits = std::numeric_limits<uint32_t>::max();
};

struct PreparedExecution::Impl::PendingAliasGroupAllocation {
    template <typename Object> struct Peer {
        DescriptorBinding binding;
        Object object;
        vk::DeviceSize offset;
    };
    std::vector<Peer<vk::raii::TensorARM>> tensors;
    std::vector<Peer<vk::raii::Buffer>> buffers;
    std::vector<Peer<vk::raii::Image>> images;

    // Shared allocation requirements
    uint32_t memoryTypeBits = std::numeric_limits<uint32_t>::max();
    vk::DeviceSize memorySize = 0;

    bool empty() const noexcept { return tensors.empty() && buffers.empty() && images.empty(); }
};

namespace {

/*******************************************************************************
 * Internal helpers
 *******************************************************************************/

template <typename Container, typename Predicate>
std::optional<std::reference_wrapper<const typename Container::value_type>> findRefIf(const Container &container,
                                                                                      Predicate predicate) {
    const auto it = std::find_if(container.begin(), container.end(), predicate);
    if (it == container.end()) {
        return std::nullopt;
    }
    return std::cref(*it);
}

void waitForFence(const vk::raii::Device &device, const vk::raii::Fence &fence) {
    const auto result = device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max());
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("vkWaitForFences failed with VkResult " +
                                 std::to_string(static_cast<int32_t>(result)));
    }
}

vk::AccessFlags2 resourceAccessMask(ResourceAccess access, vk::AccessFlags2 readAccess, vk::AccessFlags2 writeAccess) {
    switch (access) {
    case ResourceAccess::Read:
        return readAccess;
    case ResourceAccess::Write:
        return writeAccess;
    case ResourceAccess::ReadWrite:
        return readAccess | writeAccess;
    }
    return {};
}

void recordPipelineBarrier(const vk::raii::Device &device, vk::CommandBuffer commandBuffer,
                           const vk::DependencyInfo &dependencyInfo) {
    const auto *const dispatcher = device.getDispatcher();
    if (!dispatcher->vkCmdPipelineBarrier2) {
        throw std::runtime_error("vkCmdPipelineBarrier2 is not available");
    }
    dispatcher->vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(commandBuffer),
                                      reinterpret_cast<const VkDependencyInfo *>(&dependencyInfo));
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

vk::ImageLayout requiredImageLayout(const Workload &workload, const DescriptorBinding &descBinding) {
    return workloadImpl(workload).resources.at(descBinding.resourceIndex).requiredImageLayout();
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
void bindResourceMemory(const ContextView &contextView, RuntimeResource &resource, BoundMemoryInfo memory) {
    const auto requirements = resourceMemoryRequirements(contextView, resource);
    if (memory.size < requirements.size) {
        throw std::runtime_error("Bound memory is too small for a resource");
    }
    if (requirements.alignment != 0 && memory.offset % requirements.alignment != 0) {
        throw std::runtime_error("Bound memory offset does not satisfy resource alignment");
    }
    if (memory.offset > std::numeric_limits<vk::DeviceSize>::max() - memory.size) {
        throw std::runtime_error("Bound memory range overflows");
    }

    if constexpr (std::is_same_v<std::decay_t<RuntimeResource>, vk::raii::TensorARM>) {
        contextView.device.get().bindTensorMemoryARM(
            vk::BindTensorMemoryInfoARM(*resource, memory.memory, memory.offset));
    } else {
        resource.bindMemory(memory.memory, memory.offset);
    }
}

vk::raii::DeviceMemory allocateDeviceLocalMemory(const ContextView &contextView,
                                                 const vk::MemoryRequirements &memoryRequirements) {
    const auto memoryType = detail::findMemoryType(contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits,
                                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
    return {contextView.device.get(), vk::MemoryAllocateInfo(memoryRequirements.size, memoryType)};
}

/*******************************************************************************
 * Binding validation
 *******************************************************************************/

DescriptorBinding descBindingForResource(const Workload &workload, uint32_t resourceIndex) {
    const auto &workloadState = workloadImpl(workload);
    const auto &resource = workloadState.resources.at(resourceIndex);
    if (!resource.descriptorType.has_value()) {
        throw std::runtime_error("Public resource has no descriptor type");
    }
    return {resourceIndex, 0, 0, workloadState.resourceAccess(resourceIndex)};
}

void validateBoundMemoryInfo(const Workload &workload, uint32_t resourceIndex, BoundMemoryInfo boundMemInfo) {
    const auto &resource = workloadImpl(workload).resources.at(resourceIndex);
    if ((resource.requiresBoundMemoryInfo || resource.aliasGroupId.has_value()) && boundMemInfo.memory == nullptr) {
        throw std::runtime_error("Bound memory information is required for workload resource " +
                                 std::to_string(resourceIndex));
    }
}

template <typename BoundResources>
const typename BoundResources::value_type *findBoundResource(const BoundResources &boundResources,
                                                             uint32_t resourceIndex,
                                                             std::optional<ResourceKind> requiredKind = std::nullopt) {
    const auto boundResource = findRefIf(boundResources, [resourceIndex](const auto &resource) {
        return resource.descBinding.resourceIndex == resourceIndex;
    });
    if (!boundResource.has_value() && requiredKind.has_value()) {
        throw std::runtime_error("No " + std::string(detail::resourceKindName(*requiredKind)) +
                                 " bound for workload resource " + std::to_string(resourceIndex));
    }
    return boundResource.has_value() ? &boundResource->get() : nullptr;
}

/*******************************************************************************
 * Intermediate object creation
 *******************************************************************************/

void planRuntimeResourceAllocations(const Workload &workload,
                                    std::vector<DescriptorBinding> &unaliasedIntermediateDescBindings,
                                    std::map<uint32_t, std::vector<DescriptorBinding>> &aliasGroups) {
    std::vector<uint32_t> plannedResourceIndices;
    const auto &workloadState = workloadImpl(workload);
    for (const auto &executable : workloadState.executables) {
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

    std::vector<DescriptorBinding> unaliasedIntermediateDescBindings;
    std::map<uint32_t, std::vector<DescriptorBinding>> aliasGroups;
    planRuntimeResourceAllocations(sessionImpl.workload, unaliasedIntermediateDescBindings, aliasGroups);
    resolveUnaliasedIntermediateAllocations(unaliasedIntermediateDescBindings);
    resolveAliasGroups(aliasGroups, bindingState.aliasPeerOffsetsByResourceIndex);

    createDescriptorSets();
    writeDescriptors();
    buildExecutionBarriers();
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
    for (uint32_t index = 0; index < sessionImpl.executableStates.size(); ++index) {
        const auto requiredSize =
            sessionImpl.executableStates[index].compiledExecutable.get().requiredPushConstantSize();
        const auto payload = bindingState.pushConstantsByExecutableIndex.find(index);
        const auto providedSize =
            payload == bindingState.pushConstantsByExecutableIndex.end() ? 0 : payload->second.size();
        if (providedSize != requiredSize) {
            throw std::runtime_error("BindingSet push constant size does not match executable " +
                                     std::to_string(index) + " requirements");
        }
    }
    pushConstantsByExecutableIndex = bindingState.pushConstantsByExecutableIndex;
}

void PreparedExecution::Impl::addBoundTensor(TensorBindingInfo tensorBindingInfo, DescriptorBinding descBinding) {
    if (findBoundResource(boundTensors, descBinding.resourceIndex)) {
        return;
    }
    const auto &resource = workloadImpl(sessionImpl.workload).resources.at(descBinding.resourceIndex);
    const vk::TensorViewCreateInfoARM viewCreateInfo({}, tensorBindingInfo.tensor, resource.format);
    boundTensors.push_back({descBinding, tensorBindingInfo.tensor,
                            vk::raii::TensorViewARM(sessionImpl.contextView.device.get(), viewCreateInfo),
                            tensorBindingInfo.memory});
}

void PreparedExecution::Impl::addBoundBuffer(BufferBindingInfo bufferBindingInfo, DescriptorBinding descBinding) {
    if (findBoundResource(boundBuffers, descBinding.resourceIndex)) {
        return;
    }
    boundBuffers.push_back({descBinding, bufferBindingInfo.buffer, bufferBindingInfo.memory});
}

void PreparedExecution::Impl::addBoundImage(ImageBindingInfo imageBindingInfo, DescriptorBinding descBinding) {
    if (findBoundResource(boundImages, descBinding.resourceIndex)) {
        return;
    }
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
    const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);

    auto subresourceRange = imageBindingInfo.subresourceRange;
    if (!subresourceRange.aspectMask) {
        subresourceRange = resource.requiredImageSubresourceRange();
    }

    vk::raii::ImageView ownedImageView(nullptr);
    vk::ImageView imageView = imageBindingInfo.imageView;
    if (imageView == nullptr) {
        const auto viewType = detail::imageViewType(detail::imageMetadata(resource).imageType);
        const vk::ImageViewCreateInfo viewCreateInfo({}, imageBindingInfo.image, viewType, resource.format, {},
                                                     subresourceRange);
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
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    for (const auto &descBinding : descBindings) {
        const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
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

void PreparedExecution::Impl::resolveAliasGroups(const std::map<uint32_t, std::vector<DescriptorBinding>> &aliasGroups,
                                                 const std::map<uint32_t, vk::DeviceSize> &aliasPeerOffsets) {
    for (const auto &aliasGroup : aliasGroups) {
        const auto &resources = workloadImpl(sessionImpl.workload).resources;
        const bool createsPeers =
            std::any_of(aliasGroup.second.begin(), aliasGroup.second.end(), [&resources](const auto &binding) {
                return resources.at(binding.resourceIndex).role == Resource::Role::Intermediate;
            });
        const auto existingBindings = findExistingAliasGroupBindings(aliasGroup.first, createsPeers);

        PendingAliasGroupAllocation pendingAllocation;
        pendingAllocation.memoryTypeBits = existingBindings.memoryTypeBits;
        for (const auto &descBinding : aliasGroup.second) {
            resolveAliasGroupBinding(descBinding, existingBindings, pendingAllocation, aliasPeerOffsets);
        }

        if (!pendingAllocation.empty()) {
            // All peers in the group have been created and validated before any bind.
            auto aliasMemory = existingBindings.boundMemory;
            if (!aliasMemory.has_value()) {
                auto memory = allocateAliasGroupMemory(pendingAllocation);
                aliasMemory = runtimeResources.adoptMemory(std::move(memory), pendingAllocation.memorySize);
            }
            bindAliasGroupMemory(pendingAllocation, *aliasMemory);
            adoptAliasGroupResources(pendingAllocation, *aliasMemory);
        }
    }
}

// Resolve unaliased intermediates.
void PreparedExecution::Impl::resolveUnaliasedIntermediateTensor(const DescriptorBinding &descBinding) {
    const auto tiling = detail::toTensorTiling(sessionImpl.resourceTilings.at(descBinding.resourceIndex));
    auto tensor = createIntermediateTensor(sessionImpl.workload, sessionImpl.contextView, descBinding.resourceIndex,
                                           tiling, sessionImpl.tensorStrides(descBinding.resourceIndex));
    const auto memoryRequirements = resourceMemoryRequirements(sessionImpl.contextView, tensor);
    auto memory = allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
    bindResourceMemory(sessionImpl.contextView, tensor, {*memory, 0, memoryRequirements.size});
    const auto tensorBindingInfo =
        runtimeResources.adoptTensor(std::move(tensor), std::move(memory), memoryRequirements.size);
    addBoundTensor(tensorBindingInfo, descBinding);
}

void PreparedExecution::Impl::resolveUnaliasedIntermediateBuffer(const DescriptorBinding &descBinding) {
    auto buffer = createIntermediateBuffer(sessionImpl.workload, sessionImpl.contextView, descBinding.resourceIndex);
    const auto memoryRequirements = resourceMemoryRequirements(sessionImpl.contextView, buffer);
    auto memory = allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
    bindResourceMemory(sessionImpl.contextView, buffer, {*memory, 0, memoryRequirements.size});
    const auto bufferBindingInfo =
        runtimeResources.adoptBuffer(std::move(buffer), std::move(memory), memoryRequirements.size);
    addBoundBuffer(bufferBindingInfo, descBinding);
}

void PreparedExecution::Impl::resolveUnaliasedIntermediateImage(const DescriptorBinding &descBinding) {
    const auto tiling = detail::toImageTiling(sessionImpl.resourceTilings.at(descBinding.resourceIndex));
    auto image = createIntermediateImage(sessionImpl.workload, sessionImpl.contextView, descBinding.resourceIndex,
                                         false, tiling);
    const auto memoryRequirements = resourceMemoryRequirements(sessionImpl.contextView, image);
    auto memory = allocateDeviceLocalMemory(sessionImpl.contextView, memoryRequirements);
    const BoundMemoryInfo bindingMemory{*memory, 0, memoryRequirements.size};
    bindResourceMemory(sessionImpl.contextView, image, bindingMemory);
    const auto imageBindingInfo =
        runtimeResources.adoptImage(std::move(image), std::move(memory), memoryRequirements.size, bindingMemory);
    addBoundImage(imageBindingInfo, descBinding);
}

// Resolve alias groups.
PreparedExecution::Impl::ExistingAliasGroupBindings
PreparedExecution::Impl::findExistingAliasGroupBindings(uint32_t aliasGroupId, bool createsPeers) const {
    ExistingAliasGroupBindings existingBindings;
    const auto &workload = sessionImpl.workload;
    const auto belongsToGroup = [&workload, aliasGroupId](const auto &boundResource) {
        return workloadImpl(workload).resources.at(boundResource.descBinding.resourceIndex).aliasGroupId ==
               aliasGroupId;
    };
    const auto rememberBoundMemory = [&, aliasGroupId](BoundMemoryInfo memory) {
        if (memory.memory == nullptr) {
            throw std::runtime_error("Manually bound aliases must provide memory for alias group " +
                                     std::to_string(aliasGroupId));
        }
        if (existingBindings.boundMemory.has_value()) {
            const auto &previous = *existingBindings.boundMemory;
            if (previous.memory != memory.memory || previous.offset != memory.offset || previous.size != memory.size) {
                throw std::runtime_error("Conflicting backing regions for alias group " + std::to_string(aliasGroupId));
            }
        }
        existingBindings.boundMemory = memory;
    };
    const auto rememberRequirements = [&](auto handle, uint32_t resourceIndex) {
        if (!createsPeers) {
            return;
        }
        const auto requirements = aliasMemoryRequirements(sessionImpl.contextView, handle);
        const auto &memory = requirements.template get<vk::MemoryRequirements2>().memoryRequirements;
        (void)detail::validateAliasMemoryBinding(
            resourceIndex, *existingBindings.boundMemory, 0, memory,
            requirements.template get<vk::MemoryDedicatedRequirements>().requiresDedicatedAllocation);
        existingBindings.memoryTypeBits &= memory.memoryTypeBits;
        if (existingBindings.memoryTypeBits == 0) {
            throw std::runtime_error("Alias resources have no common memory type");
        }
    };
    for (const auto &bound : boundTensors) {
        if (belongsToGroup(bound)) {
            rememberBoundMemory(bound.memory);
            rememberRequirements(bound.tensor, bound.descBinding.resourceIndex);
            if (!existingBindings.boundTensor.has_value()) {
                existingBindings.boundTensor = TensorBindingInfo{bound.tensor, bound.memory};
            }
        }
    }
    for (const auto &bound : boundBuffers) {
        if (belongsToGroup(bound)) {
            rememberBoundMemory(bound.memory);
            rememberRequirements(bound.buffer, bound.descBinding.resourceIndex);
            if (!existingBindings.boundBuffer.has_value()) {
                existingBindings.boundBuffer = BufferBindingInfo{bound.buffer, bound.memory};
            }
        }
    }
    for (const auto &bound : boundImages) {
        if (belongsToGroup(bound)) {
            rememberBoundMemory(bound.memory);
            rememberRequirements(bound.image, bound.descBinding.resourceIndex);
            if (!existingBindings.boundImage.has_value()) {
                existingBindings.boundImage = ImageBindingInfo{bound.image,   bound.memory, bound.imageView,
                                                               bound.sampler, bound.layout, bound.subresourceRange};
            }
        }
    }

    return existingBindings;
}

void PreparedExecution::Impl::resolveAliasGroupBinding(const DescriptorBinding &descBinding,
                                                       const ExistingAliasGroupBindings &existingBindings,
                                                       PendingAliasGroupAllocation &pendingAllocation,
                                                       const std::map<uint32_t, vk::DeviceSize> &aliasPeerOffsets) {
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
    const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
    if (resource.role != Resource::Role::Intermediate) {
        addExistingAliasBinding(descBinding, descriptorType, existingBindings);
        return;
    }

    const auto peerOffset = aliasPeerOffsets.find(descBinding.resourceIndex);
    if (peerOffset != aliasPeerOffsets.end() && !existingBindings.boundMemory.has_value()) {
        throw std::runtime_error("Alias peer offsets require external backing");
    }
    const auto addPendingResource = [&](auto object, auto &objects) {
        const auto chain = aliasMemoryRequirements(sessionImpl.contextView, *object);
        const auto &requirements = chain.template get<vk::MemoryRequirements2>().memoryRequirements;
        const bool dedicated = chain.template get<vk::MemoryDedicatedRequirements>().requiresDedicatedAllocation;
        vk::DeviceSize offset = 0;
        if (existingBindings.boundMemory.has_value()) {
            const auto byteOffset = peerOffset == aliasPeerOffsets.end() ? 0 : peerOffset->second;
            offset = detail::validateAliasMemoryBinding(descBinding.resourceIndex, *existingBindings.boundMemory,
                                                        byteOffset, requirements, dedicated);
        } else {
            if (dedicated) {
                throw std::runtime_error("Alias peer requires a dedicated allocation");
            }
            pendingAllocation.memorySize = std::max(pendingAllocation.memorySize, requirements.size);
        }
        pendingAllocation.memoryTypeBits &= requirements.memoryTypeBits;
        if (pendingAllocation.memoryTypeBits == 0) {
            throw std::runtime_error("Alias resources have no common memory type");
        }
        objects.push_back({descBinding, std::move(object), offset});
    };

    switch (descriptorType) {
    case vk::DescriptorType::eTensorARM: {
        const auto tiling = detail::toTensorTiling(sessionImpl.resourceTilings.at(descBinding.resourceIndex));
        addPendingResource(createIntermediateTensor(sessionImpl.workload, sessionImpl.contextView,
                                                    descBinding.resourceIndex, tiling,
                                                    sessionImpl.tensorStrides(descBinding.resourceIndex)),
                           pendingAllocation.tensors);
        return;
    }
    case vk::DescriptorType::eStorageBuffer:
        addPendingResource(
            createIntermediateBuffer(sessionImpl.workload, sessionImpl.contextView, descBinding.resourceIndex),
            pendingAllocation.buffers);
        return;
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage: {
        const auto tiling = detail::toImageTiling(sessionImpl.resourceTilings.at(descBinding.resourceIndex));
        addPendingResource(createIntermediateImage(sessionImpl.workload, sessionImpl.contextView,
                                                   descBinding.resourceIndex, true, tiling),
                           pendingAllocation.images);
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
        if (findRefIf(boundTensors, [&descBinding](const auto &bound) {
                return bound.descBinding.resourceIndex == descBinding.resourceIndex;
            }).has_value()) {
            return;
        }
        if (!existingBindings.boundTensor.has_value()) {
            throwNoAlias("tensor");
        }
        addBoundTensor(*existingBindings.boundTensor, descBinding);
        return;
    }
    case vk::DescriptorType::eStorageBuffer: {
        if (findRefIf(boundBuffers, [&descBinding](const auto &bound) {
                return bound.descBinding.resourceIndex == descBinding.resourceIndex;
            }).has_value()) {
            return;
        }
        if (!existingBindings.boundBuffer.has_value()) {
            throwNoAlias("buffer");
        }
        addBoundBuffer(*existingBindings.boundBuffer, descBinding);
        return;
    }
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage: {
        if (findRefIf(boundImages, [&descBinding](const auto &bound) {
                return bound.descBinding.resourceIndex == descBinding.resourceIndex;
            }).has_value()) {
            return;
        }
        if (!existingBindings.boundImage.has_value()) {
            throwNoAlias("image");
        }
        addBoundImage(*existingBindings.boundImage, descBinding);
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
                                                   BoundMemoryInfo aliasMemory) const {
    const auto bind = [&](auto &peers) {
        for (auto &peer : peers) {
            const auto remainingSize = aliasMemory.size - (peer.offset - aliasMemory.offset);
            bindResourceMemory(sessionImpl.contextView, peer.object, {aliasMemory.memory, peer.offset, remainingSize});
        }
    };
    bind(pendingAllocation.tensors);
    bind(pendingAllocation.buffers);
    bind(pendingAllocation.images);
}

void PreparedExecution::Impl::adoptAliasGroupResources(PendingAliasGroupAllocation &pendingAllocation,
                                                       BoundMemoryInfo aliasMemory) {
    for (auto &peer : pendingAllocation.tensors) {
        const auto tensorBindingInfo = runtimeResources.adoptTensor(std::move(peer.object), aliasMemory);
        addBoundTensor(tensorBindingInfo, peer.binding);
    }

    for (auto &peer : pendingAllocation.buffers) {
        const auto bufferBindingInfo = runtimeResources.adoptBuffer(std::move(peer.object), aliasMemory);
        addBoundBuffer(bufferBindingInfo, peer.binding);
    }

    for (auto &peer : pendingAllocation.images) {
        const auto imageBindingInfo = runtimeResources.adoptImage(std::move(peer.object), aliasMemory);
        addBoundImage(imageBindingInfo, peer.binding);
    }
}

/*******************************************************************************
 * Descriptor setup
 *******************************************************************************/

void PreparedExecution::Impl::createDescriptorSets() {
    descriptorSetStates.reserve(sessionImpl.executableStates.size());
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    for (std::size_t executableIndex = 0; executableIndex < sessionImpl.executableStates.size(); ++executableIndex) {
        auto &descriptors = descriptorSetStates.emplace_back();
        std::map<vk::DescriptorType, uint32_t> descriptorCounts;
        const auto &executable = workloadState.executables.at(executableIndex);
        for (const auto &descBinding : executable.bindings) {
            ++descriptorCounts[workloadState.descriptorTypeForBinding(descBinding)];
        }
        std::vector<vk::DescriptorPoolSize> poolSizes;
        poolSizes.reserve(descriptorCounts.size());
        for (const auto &[type, count] : descriptorCounts) {
            poolSizes.emplace_back(type, count);
        }
        const auto &compiledExecutable = sessionImpl.executableStates[executableIndex].compiledExecutable.get();
        descriptors.descriptorPool = vk::raii::DescriptorPool(
            sessionImpl.contextView.device.get(),
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
             static_cast<uint32_t>(compiledExecutable.descriptorSetLayouts.size()), poolSizes});
        const auto descriptorSetLayouts = detail::rawDescriptorSetLayouts(compiledExecutable.descriptorSetLayouts);
        const vk::DescriptorSetAllocateInfo allocateInfo(*descriptors.descriptorPool, descriptorSetLayouts);
        descriptors.descriptorSets = sessionImpl.contextView.device.get().allocateDescriptorSets(allocateInfo);
    }
}

void PreparedExecution::Impl::writeDescriptors() const {
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    for (std::size_t executableIndex = 0; executableIndex < sessionImpl.executableStates.size(); ++executableIndex) {
        const auto &descriptorSets = descriptorSetStates[executableIndex].descriptorSets;
        const auto &executable = workloadState.executables.at(executableIndex);

        for (const auto &descBinding : executable.bindings) {
            const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
            switch (descriptorType) {
            case vk::DescriptorType::eTensorARM: {
                const auto *tensor = findBoundResource(boundTensors, descBinding.resourceIndex, ResourceKind::Tensor);
                const auto tensorView = *tensor->tensorView;
                const vk::WriteDescriptorSetTensorARM tensorInfo(1, &tensorView);
                const vk::WriteDescriptorSet write(*descriptorSets[descBinding.set], descBinding.binding, 0, 1,
                                                   descriptorType, nullptr, nullptr, nullptr, &tensorInfo);
                sessionImpl.contextView.device.get().updateDescriptorSets(write, nullptr);
                break;
            }
            case vk::DescriptorType::eStorageBuffer: {
                const auto *buffer =
                    findBoundResource(boundBuffers, descBinding.resourceIndex, ResourceKind::StorageBuffer);
                const vk::DescriptorBufferInfo bufferInfo(buffer->buffer, 0, vk::WholeSize);
                const vk::WriteDescriptorSet write(*descriptorSets[descBinding.set], descBinding.binding, 0, 1,
                                                   descriptorType, nullptr, &bufferInfo);
                sessionImpl.contextView.device.get().updateDescriptorSets(write, nullptr);
                break;
            }
            case vk::DescriptorType::eCombinedImageSampler:
            case vk::DescriptorType::eStorageImage: {
                const auto *image = findBoundResource(boundImages, descBinding.resourceIndex, ResourceKind::Image);
                const vk::DescriptorImageInfo imageInfo(
                    descriptorType == vk::DescriptorType::eCombinedImageSampler ? image->sampler : vk::Sampler(),
                    image->imageView, requiredImageLayout(sessionImpl.workload, descBinding));
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

void PreparedExecution::Impl::buildExecutionBarriers() {
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    auto state = buildSynchronizationState();
    executableBarriers.reserve(workloadState.executables.size() + 1);
    for (const auto &executable : workloadState.executables) {
        const auto uses = state.collectResourceUses(executable);
        executableBarriers.push_back(state.buildExecutableBarriers(executable, uses));
    }
    executableBarriers.push_back(state.buildExitBarriers());
}

PreparedExecution::Impl::SynchronizationState PreparedExecution::Impl::buildSynchronizationState() const {
    const auto &workloadState = workloadImpl(sessionImpl.workload);
    SynchronizationState state;
    state.resources.resize(workloadState.resources.size());
    state.slots.reserve(workloadState.resources.size());
    state.images.reserve(boundImages.size());

    std::map<uint32_t, uint32_t> synchronizationSlotByAliasGroup;
    for (uint32_t resourceIndex = 0; resourceIndex < workloadState.resources.size(); ++resourceIndex) {
        const auto &resource = workloadState.resources[resourceIndex];
        auto &info = state.resources[resourceIndex];
        info.isOptimalTensor = resource.descriptorType == vk::DescriptorType::eTensorARM &&
                               sessionImpl.resourceTilings.at(resourceIndex) == detail::ResourceTiling::Optimal;
        if (resource.aliasGroupId.has_value()) {
            const auto [it, inserted] = synchronizationSlotByAliasGroup.try_emplace(
                *resource.aliasGroupId, static_cast<uint32_t>(state.slots.size()));
            if (inserted) {
                state.slots.emplace_back().isAliasGroup = true;
            }
            info.synchronizationSlotIndex = it->second;
        } else {
            info.synchronizationSlotIndex = static_cast<uint32_t>(state.slots.size());
            state.slots.emplace_back();
        }
    }

    // Descriptor setup has already checked the handles. Resolve image identity and
    // layout metadata once, including images used only as external alias backing.
    for (const auto &boundImage : boundImages) {
        const auto resourceIndex = boundImage.descBinding.resourceIndex;
        auto &info = state.resources.at(resourceIndex);
        const auto slotIndex = info.synchronizationSlotIndex;
        const auto descriptorLayout = requiredImageLayout(sessionImpl.workload, boundImage.descBinding);
        const auto sameImageState = [&boundImage](const SynchronizationState::Image &imageState) {
            return imageState.image == boundImage.image && imageState.subresourceRange == boundImage.subresourceRange;
        };
        const auto existing = std::find_if(state.images.begin(), state.images.end(), sameImageState);
        uint32_t imageStateIndex = 0;
        if (existing == state.images.end()) {
            imageStateIndex = static_cast<uint32_t>(state.images.size());
            state.images.push_back({boundImage.image, boundImage.subresourceRange, boundImage.layout,
                                    boundImage.layout.value_or(descriptorLayout), descriptorLayout, slotIndex});
        } else {
            if (existing->descriptorLayout != descriptorLayout || existing->synchronizationSlotIndex != slotIndex) {
                throw std::runtime_error("The same bound image requires incompatible layout or alias group state");
            }
            imageStateIndex = static_cast<uint32_t>(std::distance(state.images.begin(), existing));
            if (!existing->initialLayout.has_value()) {
                existing->initialLayout = boundImage.layout;
                if (boundImage.layout.has_value()) {
                    existing->currentLayout = *boundImage.layout;
                }
            } else if (boundImage.layout.has_value() && existing->initialLayout != boundImage.layout) {
                throw std::runtime_error("Conflicting layouts provided for the same bound image");
            }
        }

        info.imageStateIndex = imageStateIndex;
        auto &slotImages = state.slots.at(slotIndex).imageStateIndices;
        if (std::find(slotImages.begin(), slotImages.end(), imageStateIndex) == slotImages.end()) {
            slotImages.push_back(imageStateIndex);
        }
    }
    return state;
}

std::vector<PreparedExecution::Impl::ResourceUse>
PreparedExecution::Impl::SynchronizationState::collectResourceUses(const detail::Executable &executable) const {
    std::vector<ResourceUse> uses;
    uses.reserve(executable.bindings.size());
    const auto readAccess = detail::readAccess(executable.type);
    const auto writeAccess = detail::writeAccess(executable.type);
    for (const auto &binding : executable.bindings) {
        const auto &resource = resources.at(binding.resourceIndex);
        const auto existingUse = std::find_if(uses.begin(), uses.end(), [&](const auto &use) {
            return resources[use.resourceIndex].synchronizationSlotIndex == resource.synchronizationSlotIndex;
        });
        const auto accessMask = resourceAccessMask(binding.access, readAccess, writeAccess);
        const bool writes = binding.access != ResourceAccess::Read;
        if (existingUse != uses.end()) {
            // One dependency covers the group's memory. Multiple peers may be
            // accessed together when they require the same image layout state.
            const auto &existingResource = resources[existingUse->resourceIndex];
            if (existingResource.imageStateIndex != resource.imageStateIndex ||
                existingResource.isOptimalTensor != resource.isOptimalTensor) {
                throw std::runtime_error("Executable accesses alias peers requiring different image layout states");
            }
            existingUse->accessMask |= accessMask;
            existingUse->writes |= writes;
        } else {
            uses.push_back({binding.resourceIndex, accessMask, writes});
        }
    }
    return uses;
}

PreparedExecution::Impl::BarrierBatch PreparedExecution::Impl::SynchronizationState::buildExitBarriers() {
    // Restore reusable entry layouts so prebuilt barriers can be replayed.
    // Preinitialized entries are one-shot; Undefined entries discard on replay.
    BarrierBatch barriers;
    for (uint32_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        const auto &image = images[imageIndex];
        const auto entryLayout = image.initialLayout.value_or(image.descriptorLayout);
        const bool reusableEntryLayout =
            entryLayout != vk::ImageLayout::eUndefined && entryLayout != vk::ImageLayout::ePreinitialized;
        const auto exitLayout = reusableEntryLayout ? entryLayout : image.descriptorLayout;
        if (!image.used || image.currentLayout == exitLayout) {
            continue;
        }
        // The caller chooses the next consumer.
        const auto dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        const auto dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
        ensureImageLayout(barriers, imageIndex, exitLayout, dstStageMask, dstAccessMask);
        const auto &slot = slots.at(image.synchronizationSlotIndex);
        // The image barrier restores the layout, but its access scopes cover only
        // this image. Also make pending writes through alias peers visible.
        if (slot.pendingWrites) {
            barriers.memoryBarriers.emplace_back(slot.pendingStageMask, slot.pendingAccessMask, dstStageMask,
                                                 dstAccessMask);
        }
    }
    return barriers;
}

bool PreparedExecution::Impl::SynchronizationState::ensureImageLayout(BarrierBatch &barriers, uint32_t imageIndex,
                                                                      vk::ImageLayout layout,
                                                                      vk::PipelineStageFlags2 dstStageMask,
                                                                      vk::AccessFlags2 dstAccessMask) {
    auto &image = images.at(imageIndex);
    const bool initialBarrier = !image.used && image.initialLayout.has_value();
    const bool layoutChanged = image.currentLayout != layout;
    image.used = true;
    if (!layoutChanged && !initialBarrier) {
        return false;
    }
    const auto &slot = slots.at(image.synchronizationSlotIndex);
    if (layoutChanged && slot.used && !slot.lastImplicitBarrier) {
        throw std::runtime_error("Image alias layout transition follows an executable with implicit barriers disabled");
    }

    vk::ImageMemoryBarrier2 barrier;
    if (slot.pendingStageMask) {
        barrier.srcStageMask = slot.pendingStageMask;
        barrier.srcAccessMask = slot.pendingAccessMask;
    } else if (!slot.used) {
        // Discarding contents does not order earlier accesses.
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    }
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = image.currentLayout;
    barrier.newLayout = layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = image.subresourceRange;
    barriers.imageBarriers.push_back(barrier);
    image.currentLayout = layout;
    return true;
}

bool PreparedExecution::Impl::SynchronizationState::prepareImageAccess(BarrierBatch &barriers,
                                                                       const ResourceInfo &resource,
                                                                       vk::PipelineStageFlags2 stageMask,
                                                                       vk::AccessFlags2 accessMask) {
    if (resource.imageStateIndex.has_value()) {
        const auto imageIndex = *resource.imageStateIndex;
        return ensureImageLayout(barriers, imageIndex, images[imageIndex].descriptorLayout, stageMask, accessMask);
    }
    bool imageBarrierEmitted = false;
    if (resource.isOptimalTensor) {
        for (const auto imageIndex : slots[resource.synchronizationSlotIndex].imageStateIndices) {
            imageBarrierEmitted |=
                ensureImageLayout(barriers, imageIndex, vk::ImageLayout::eTensorAliasingARM, stageMask, accessMask);
        }
    }
    return imageBarrierEmitted;
}

void PreparedExecution::Impl::SynchronizationState::Slot::recordUse(const ResourceUse &use,
                                                                    vk::PipelineStageFlags2 stageMask,
                                                                    bool implicitBarrier, bool imageBarrierEmitted) {
    used = true;
    lastImplicitBarrier = implicitBarrier;
    // Retain writer scope across readers, which may execute at different stages.
    if (use.writes || imageBarrierEmitted) {
        pendingStageMask = {};
        pendingAccessMask = {};
        pendingWrites = false;
    }
    if (implicitBarrier) {
        pendingStageMask |= stageMask;
        pendingAccessMask |= use.accessMask;
        pendingWrites |= use.writes || imageBarrierEmitted;
        if (imageBarrierEmitted) {
            // Layout transitions may write even for read-only executables. Keep
            // the same conservative write scope for same-layout entry barriers.
            pendingAccessMask |= vk::AccessFlagBits2::eMemoryWrite;
        }
    }
}

PreparedExecution::Impl::BarrierBatch
PreparedExecution::Impl::SynchronizationState::buildExecutableBarriers(const detail::Executable &executable,
                                                                       const std::vector<ResourceUse> &uses) {
    BarrierBatch barriers;
    const auto stageMask = detail::pipelineStage(executable.type);
    for (const auto &use : uses) {
        const auto &resource = resources.at(use.resourceIndex);
        auto &slot = slots.at(resource.synchronizationSlotIndex);

        const bool imageBarrierEmitted = prepareImageAccess(barriers, resource, stageMask, use.accessMask);
        const bool memoryDependency = slot.pendingStageMask && (slot.pendingWrites || use.writes);
        const bool externalAliasDependency = slot.isAliasGroup && !slot.used && imageBarrierEmitted;
        if (memoryDependency || externalAliasDependency) {
            vk::MemoryBarrier2 memoryBarrier;
            // Include writes through externally managed aliases.
            memoryBarrier.srcStageMask = slot.used ? slot.pendingStageMask : vk::PipelineStageFlagBits2::eAllCommands;
            memoryBarrier.srcAccessMask = slot.used
                                              ? slot.pendingAccessMask
                                              : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
            memoryBarrier.dstStageMask = stageMask;
            memoryBarrier.dstAccessMask = use.accessMask;
            barriers.memoryBarriers.push_back(memoryBarrier);
        }
        slot.recordUse(use, stageMask, executable.implicitBarrier, imageBarrierEmitted);
    }
    return barriers;
}

void PreparedExecution::Impl::BarrierBatch::record(const vk::raii::Device &device,
                                                   vk::CommandBuffer commandBuffer) const {
    if (memoryBarriers.empty() && imageBarriers.empty()) {
        return;
    }

    vk::DependencyInfo dependencyInfo;
    dependencyInfo.memoryBarrierCount = static_cast<uint32_t>(memoryBarriers.size());
    dependencyInfo.pMemoryBarriers = memoryBarriers.data();
    dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    dependencyInfo.pImageMemoryBarriers = imageBarriers.data();

    recordPipelineBarrier(device, commandBuffer, dependencyInfo);
}

/*******************************************************************************
 * Recording and submission
 *******************************************************************************/

void PreparedExecution::Impl::record(vk::CommandBuffer commandBuffer, const RecordOptions &options) {
    if (commandBuffer == nullptr) {
        throw std::runtime_error("PreparedExecution::record() requires a valid command buffer");
    }

    const auto &sessionState = sessionImpl;
    const auto *const dispatcher = sessionState.contextView.device.get().getDispatcher();
    if (!dispatcher->vkCmdBindDescriptorSets || !dispatcher->vkCmdBindPipeline || !dispatcher->vkCmdDispatch ||
        !dispatcher->vkCmdPushConstants) {
        throw std::runtime_error("Vulkan command recording functions are not available");
    }

    const auto &workloadState = workloadImpl(sessionState.workload);
    for (std::size_t executableIndex = 0; executableIndex < sessionState.executableStates.size(); ++executableIndex) {
        executableBarriers[executableIndex].record(sessionState.contextView.device.get(), commandBuffer);

        const auto &executableState = sessionState.executableStates[executableIndex];
        const auto &executable = workloadState.executables.at(executableIndex);
        const auto &compiledExecutable = executableState.compiledExecutable.get();
        const auto &descriptors = descriptorSetStates[executableIndex];
        const auto pipelineBindPoint = detail::pipelineBindPoint(executable.type);
        for (uint32_t set = 0; set < static_cast<uint32_t>(descriptors.descriptorSets.size()); ++set) {
            auto *const descriptorSet = static_cast<VkDescriptorSet>(*descriptors.descriptorSets[set]);
            dispatcher->vkCmdBindDescriptorSets(
                static_cast<VkCommandBuffer>(commandBuffer), static_cast<VkPipelineBindPoint>(pipelineBindPoint),
                static_cast<VkPipelineLayout>(*compiledExecutable.pipelineLayout), set, 1, &descriptorSet, 0, nullptr);
        }
        dispatcher->vkCmdBindPipeline(static_cast<VkCommandBuffer>(commandBuffer),
                                      static_cast<VkPipelineBindPoint>(pipelineBindPoint),
                                      static_cast<VkPipeline>(*compiledExecutable.pipeline));
        for (const auto &range : compiledExecutable.pushConstantRanges) {
            const auto &payload = pushConstantsByExecutableIndex.at(static_cast<uint32_t>(executableIndex));
            dispatcher->vkCmdPushConstants(static_cast<VkCommandBuffer>(commandBuffer),
                                           static_cast<VkPipelineLayout>(*compiledExecutable.pipelineLayout),
                                           static_cast<VkShaderStageFlags>(range.stageFlags), range.offset, range.size,
                                           payload.data() + range.offset);
        }
        const auto executableView = sessionState.workload.executable(static_cast<uint32_t>(executableIndex));
        const auto invokeExecutableHook = [&](ExecutableRecordPhase phase) {
            if (options.executableHook) {
                options.executableHook({commandBuffer, executableView, phase});
            }
        };
        invokeExecutableHook(ExecutableRecordPhase::Before);
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
        invokeExecutableHook(ExecutableRecordPhase::After);
    }
    executableBarriers.back().record(sessionState.contextView.device.get(), commandBuffer);
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

void PreparedExecution::record(vk::CommandBuffer commandBuffer, const RecordOptions &options) {
    preparedExecutionImpl().record(commandBuffer, options);
}

} // namespace mlsdk::workloadlib
