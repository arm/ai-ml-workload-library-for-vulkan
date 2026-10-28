/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "session_tiling.hpp"
#include "workload_impl.hpp"

#include <type_traits>

namespace mlsdk::workloadlib::detail {

void validateAliasTensorDescription(uint32_t resourceIndex, const Resource &resource,
                                    const std::vector<int64_t> &strides,
                                    const vk::PhysicalDeviceTensorPropertiesARM *limits = nullptr);

vk::TensorDescriptionARM tensorDescription(const Workload &workload, uint32_t resourceIndex, vk::TensorTilingARM tiling,
                                           const std::vector<int64_t> &strides);
vk::raii::TensorARM createIntermediateTensor(const Workload &workload, const ContextView &contextView,
                                             uint32_t resourceIndex, vk::TensorTilingARM tiling,
                                             const std::vector<int64_t> &strides);
vk::raii::Buffer createIntermediateBuffer(const Workload &workload, const ContextView &contextView,
                                          uint32_t resourceIndex);
vk::raii::Image createIntermediateImage(const Workload &workload, const ContextView &contextView,
                                        uint32_t resourceIndex, bool forAliasing, vk::ImageTiling tiling);
AliasPeerMemoryRequirements queryAliasPeerMemoryRequirements(const Workload &workload, const ContextView &contextView,
                                                             uint32_t resourceIndex, ResourceTiling tiling,
                                                             const std::vector<int64_t> &strides);

template <typename Handle>
vk::StructureChain<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>
aliasMemoryRequirements(const ContextView &contextView, Handle resource) {
    const auto &device = contextView.device.get();
    if constexpr (std::is_same_v<Handle, vk::TensorARM>) {
        return device.getTensorMemoryRequirementsARM<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(
            vk::TensorMemoryRequirementsInfoARM(resource));
    } else if constexpr (std::is_same_v<Handle, vk::Buffer>) {
        return device.getBufferMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(
            vk::BufferMemoryRequirementsInfo2(resource));
    } else {
        return device.getImageMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(
            vk::ImageMemoryRequirementsInfo2(resource));
    }
}

} // namespace mlsdk::workloadlib::detail
