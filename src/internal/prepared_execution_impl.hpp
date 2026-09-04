/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "binding_set_impl.hpp"
#include "session_impl.hpp"

#include "mlworkloadlib/prepared_execution.hpp"

#include <optional>
#include <vector>

namespace mlworkloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct PreparedExecution::Impl {
    using DescriptorBinding = detail::DescriptorBinding;

    /***************************************************************************
     * Bound resource records
     **************************************************************************/

    struct BoundTensor {
        DescriptorBinding descBinding;
        vk::TensorARM tensor{nullptr};
        vk::raii::TensorViewARM tensorView{nullptr};
        BoundMemoryInfo memory{};
    };

    struct BoundBuffer {
        DescriptorBinding descBinding;
        vk::Buffer buffer{nullptr};
        BoundMemoryInfo memory{};
    };

    struct BoundImage {
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

    struct DescriptorSetState {
        vk::raii::DescriptorPool descriptorPool{nullptr};
        std::vector<vk::raii::DescriptorSet> descriptorSets;
    };

    /***************************************************************************
     * Runtime resource storage
     **************************************************************************/

    struct RuntimeResourceStorage {
        // Adopt memory shared by an alias group.
        BoundMemoryInfo adoptMemory(vk::raii::DeviceMemory memory, vk::DeviceSize size);

        // Adopt resources backed by existing memory information.
        TensorBindingInfo adoptTensor(vk::raii::TensorARM tensor, BoundMemoryInfo memory = {});
        BufferBindingInfo adoptBuffer(vk::raii::Buffer buffer, BoundMemoryInfo memory = {});
        ImageBindingInfo adoptImage(vk::raii::Image image, BoundMemoryInfo memory = {});

        // Adopt resources together with private memory owned by this storage.
        TensorBindingInfo adoptTensor(vk::raii::TensorARM tensor, vk::raii::DeviceMemory memory, vk::DeviceSize size,
                                      BoundMemoryInfo bindingMemory = {});
        BufferBindingInfo adoptBuffer(vk::raii::Buffer buffer, vk::raii::DeviceMemory memory, vk::DeviceSize size,
                                      BoundMemoryInfo bindingMemory = {});
        ImageBindingInfo adoptImage(vk::raii::Image image, vk::raii::DeviceMemory memory, vk::DeviceSize size,
                                    BoundMemoryInfo bindingMemory = {});

      private:
        std::vector<vk::raii::DeviceMemory> memory_;
        std::vector<vk::raii::TensorARM> tensors_;
        std::vector<vk::raii::Buffer> buffers_;
        std::vector<vk::raii::Image> images_;
    };

    struct ExistingAliasGroupBindings;
    struct PendingAliasGroupAllocation;

    /***************************************************************************
     * Lifetime
     **************************************************************************/

    Impl(Session &sessionIn, const BindingSet &bindingsIn);

    /***************************************************************************
     * Binding setup
     **************************************************************************/

    void addBindingSetResources(const BindingSet::Impl &bindingState);
    void addBoundTensor(TensorBindingInfo tensorBindingInfo, DescriptorBinding descBinding);
    void addBoundBuffer(BufferBindingInfo bufferBindingInfo, DescriptorBinding descBinding);
    void addBoundImage(ImageBindingInfo imageBindingInfo, DescriptorBinding descBinding);
    void addBoundPushConstants(const BindingSet::Impl &bindingState);

    /***************************************************************************
     * Runtime resource allocation
     **************************************************************************/

    void resolveUnaliasedIntermediateAllocations(const std::vector<DescriptorBinding> &descBindings);
    void resolveAliasGroups(const std::map<uint32_t, std::vector<DescriptorBinding>> &aliasGroups);

    // Resolve unaliased intermediates.
    void resolveUnaliasedIntermediateTensor(const DescriptorBinding &descBinding);
    void resolveUnaliasedIntermediateBuffer(const DescriptorBinding &descBinding);
    void resolveUnaliasedIntermediateImage(const DescriptorBinding &descBinding);

    // Resolve alias groups.
    ExistingAliasGroupBindings findExistingAliasGroupBindings(uint32_t aliasGroupId) const;
    void resolveAliasGroupBinding(const DescriptorBinding &descBinding,
                                  const ExistingAliasGroupBindings &existingBindings,
                                  PendingAliasGroupAllocation &pendingAllocation);
    void addExistingAliasBinding(const DescriptorBinding &descBinding, vk::DescriptorType descriptorType,
                                 const ExistingAliasGroupBindings &existingBindings);
    vk::raii::DeviceMemory allocateAliasGroupMemory(const PendingAliasGroupAllocation &pendingAllocation) const;
    void bindAliasGroupMemory(PendingAliasGroupAllocation &pendingAllocation, vk::DeviceMemory aliasMemory) const;
    void adoptAliasGroupResources(PendingAliasGroupAllocation &pendingAllocation, BoundMemoryInfo aliasMemory);

    /***************************************************************************
     * Descriptor setup
     **************************************************************************/

    void createDescriptorSets();
    void writeDescriptors() const;

    /***************************************************************************
     * Barrier recording
     **************************************************************************/

    void insertInitialImageLayoutTransitions(vk::CommandBuffer commandBuffer);
    void insertExecutableBarrier(vk::CommandBuffer commandBuffer, const Session::Impl::ExecutableState &producer,
                                 const Session::Impl::ExecutableState &consumer) const;

    /***************************************************************************
     * Recording and submission
     **************************************************************************/

    void record(vk::CommandBuffer commandBuffer);
    void run();

    /***************************************************************************
     * Stored state
     **************************************************************************/

    Session::Impl &sessionImpl;
    std::vector<uint8_t> pushConstants;
    RuntimeResourceStorage runtimeResources;

    std::vector<BoundTensor> boundTensors;
    std::vector<BoundBuffer> boundBuffers;
    std::vector<BoundImage> boundImages;

    std::vector<DescriptorSetState> descriptorSetStates;
};

} // namespace mlworkloadlib
