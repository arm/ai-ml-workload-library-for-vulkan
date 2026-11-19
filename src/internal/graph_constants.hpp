/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "internal/utils.hpp"
#include "internal/workload_impl.hpp"

#include <stdexcept>
#include <vector>

namespace mlsdk::workloadlib::detail {

struct DataGraphPipelineCreateStorage {
    std::vector<vk::TensorDescriptionARM> resourceTensorDescriptions;
    std::vector<vk::DataGraphPipelineResourceInfoImageLayoutARM> imageLayouts;
    std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
    std::vector<vk::TensorDescriptionARM> constantTensorDescriptions;
    std::vector<vk::DataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM> sparsityInfos;
    std::vector<vk::DataGraphPipelineConstantARM> constants;
};

inline void populateDataGraphConstants(DataGraphPipelineCreateStorage &storage, const Executable &executable,
                                       const Workload &workload) {
    const auto &workloadState = workloadImpl(workload);
    const auto numConstants = static_cast<uint32_t>(executable.constantBindings.size());
    storage.constantTensorDescriptions.reserve(numConstants);
    storage.sparsityInfos.reserve(numConstants);
    storage.constants.reserve(numConstants);
    for (uint32_t constantIndex = 0; constantIndex < numConstants; ++constantIndex) {
        const auto &binding = executable.constantBindings.at(constantIndex);
        const auto workloadConstantIndex = binding.constantIndex;
        const auto &constant = workloadState.constants.at(workloadConstantIndex);
        const auto &resource = workloadState.resources.at(constant.resourceIndex);
        if (detail::tensorMetadata(resource).scaledTensorConfig) {
            throw std::runtime_error("Scaled graph constant execution is not implemented");
        }
        void *pNext = nullptr;
        if (constant.sparsityDimension < detail::Constant::unspecifiedSparsityDimension) {
            throw std::runtime_error("Graph constant has invalid sparsity dimension");
        }
        if (constant.hasSparsity()) {
            constexpr uint32_t zeroCount = 2;
            constexpr uint32_t groupSize = 4;
            storage.sparsityInfos.emplace_back(static_cast<uint32_t>(constant.sparsityDimension), zeroCount, groupSize,
                                               nullptr);
            pNext = &storage.sparsityInfos.back();
        }
        detail::validateTensorTiling(vk::TensorTilingARM::eLinear, detail::tensorMetadata(resource).usage,
                                     resource.shape, !resource.stride.empty());
        detail::validateTensorMetadata(resource.format, resource.shape, resource.stride);
        storage.constantTensorDescriptions.emplace_back(
            vk::TensorTilingARM::eLinear, resource.format, static_cast<uint32_t>(resource.shape.size()),
            resource.shape.data(), resource.stride.empty() ? nullptr : resource.stride.data(),
            vk::TensorUsageFlagBitsARM::eDataGraph, pNext);
        storage.constants.emplace_back(binding.graphConstantId, constant.payloadView.data(),
                                       &storage.constantTensorDescriptions.back());
    }
}

} // namespace mlsdk::workloadlib::detail
