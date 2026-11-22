/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "internal/utils.hpp"
#include "internal/workload_impl.hpp"

#include <array>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>

namespace mlsdk::workloadlib::detail {

struct DataGraphPipelineCreateStorage {
    std::vector<vk::TensorDescriptionARM> resourceTensorDescriptions;
    std::vector<std::array<vk::TensorDescriptionARM, 2>> scaledResourcePlaneDescriptions;
    std::vector<std::vector<int64_t>> resourceScaleStrides;
#ifdef VK_ARM_tensors2
    std::vector<vk::BlockScaledTensorDescriptionARM> scaledResourceDescriptions;
#endif
    std::vector<vk::DataGraphPipelineResourceInfoImageLayoutARM> imageLayouts;
    std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
    std::vector<vk::TensorDescriptionARM> constantTensorDescriptions;
    std::vector<std::array<vk::TensorDescriptionARM, 2>> constantPlaneDescriptions;
    std::vector<std::vector<int64_t>> constantScaleStrides;
#ifdef VK_ARM_tensors2
    std::vector<vk::BlockScaledTensorDescriptionARM> scaledConstantDescriptions;
#endif
    std::vector<vk::DataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM> sparsityInfos;
    std::vector<vk::DataGraphPipelineConstantARM> constants;
};

#ifdef VK_ARM_tensors2
inline std::vector<int64_t> validateScaledConstant(const Constant &constant, const Resource &resource) {
    const auto &metadata = tensorMetadata(resource);
    const auto &config = *metadata.scaledTensorConfig;
    constexpr uint32_t ocpMicroscalingBlockSize = 32;
    if (config.blockShape.size() != 1 || config.blockShape[0] != ocpMicroscalingBlockSize ||
        config.scaleFormat != TensorScaleFormat::E8M0) {
        throw std::runtime_error("Scaled graph constants require 32-value blocks and E8M0 scales");
    }
    const auto blockSize = config.blockShape[0];
    if (constant.sparsityDimension != Constant::unspecifiedSparsityDimension) {
        throw std::runtime_error("Scaled graph constants cannot use sparse metadata");
    }

    constexpr auto scaleFormat = vk::Format::eR8UfloatFpencodingFloat8Unsignede8M0ARM;
    const auto valueBlockExtent = vk::blockExtent(resource.format)[0];
    const auto valueElements = elementCount(resource.shape);
    const auto scaleElements = elementCount(metadata.scaleShape);
    if (resource.shape.empty() || valueElements == 0 || valueElements % blockSize != 0 ||
        valueElements / blockSize != scaleElements || (valueBlockExtent != 1 && valueBlockExtent != blockSize)) {
        throw std::runtime_error("Scaled graph constant has invalid value dimensions");
    }
    validateTensorMetadata(resource.format, resource.shape, resource.stride, /*allowPackedBlocks=*/true);
    const auto valueBlockBytes = static_cast<int64_t>(blockSize) * vk::blockSize(resource.format) / valueBlockExtent;
    auto scaleStrides = deriveScalePlaneStrides(resource.stride, valueBlockBytes);
    validateTensorMetadata(scaleFormat, metadata.scaleShape, scaleStrides);
    const auto outerElements = resource.stride.empty() ? scaleElements
                                                       : static_cast<vk::DeviceSize>(resource.shape.front()) /
                                                             (resource.shape.size() == 1 ? valueBlockExtent : 1);
    const auto outerStride = resource.stride.empty() ? static_cast<vk::DeviceSize>(valueBlockBytes)
                                                     : static_cast<vk::DeviceSize>(resource.stride.front());
    if (outerElements > constant.payloadView.size() / outerStride) {
        throw std::runtime_error("Scaled graph constant payload must contain values followed by scales");
    }
    const auto valueSize = outerElements * outerStride;
    if (valueSize % static_cast<vk::DeviceSize>(valueBlockBytes) != 0 ||
        valueSize / static_cast<vk::DeviceSize>(valueBlockBytes) != constant.payloadView.size() - valueSize) {
        throw std::runtime_error("Scaled graph constant payload must contain values followed by scales");
    }
    return scaleStrides;
}

inline void appendScaledGraphConstant(DataGraphPipelineCreateStorage &storage, uint32_t graphConstantId,
                                      const Constant &constant, const Resource &resource) {
    const auto &scaleStrides = storage.constantScaleStrides.emplace_back(validateScaledConstant(constant, resource));
    const auto &metadata = tensorMetadata(resource);
    static constexpr std::array planeTypes{vk::TensorPlaneARM::eValues, vk::TensorPlaneARM::eScales};
    auto &planes = storage.constantPlaneDescriptions.emplace_back();
    planes[0] = vk::TensorDescriptionARM(vk::TensorTilingARM::eLinear, resource.format,
                                         static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                                         resource.stride.empty() ? nullptr : resource.stride.data(),
                                         vk::TensorUsageFlagBitsARM::eDataGraph);
    planes[1] = vk::TensorDescriptionARM(
        vk::TensorTilingARM::eLinear, vk::Format::eR8UfloatFpencodingFloat8Unsignede8M0ARM,
        static_cast<uint32_t>(metadata.scaleShape.size()), metadata.scaleShape.data(),
        scaleStrides.empty() ? nullptr : scaleStrides.data(), vk::TensorUsageFlagBitsARM::eDataGraph);
    auto &description = storage.scaledConstantDescriptions.emplace_back(
        vk::TensorBlockScalingSchemeARM::ePlanarOcpMicroscaling, static_cast<uint32_t>(planes.size()),
        planeTypes.data(), planes.data());
    storage.constants.emplace_back(graphConstantId, constant.payloadView.data(), &description);
}
#endif

inline void populateDataGraphConstants(DataGraphPipelineCreateStorage &storage, const Executable &executable,
                                       const Workload &workload) {
    const auto &workloadState = workloadImpl(workload);
    const auto numConstants = static_cast<uint32_t>(executable.constantBindings.size());
    storage.constantTensorDescriptions.reserve(numConstants);
    storage.constantPlaneDescriptions.reserve(numConstants);
    storage.constantScaleStrides.reserve(numConstants);
#ifdef VK_ARM_tensors2
    storage.scaledConstantDescriptions.reserve(numConstants);
#endif
    storage.sparsityInfos.reserve(numConstants);
    storage.constants.reserve(numConstants);
    for (uint32_t constantIndex = 0; constantIndex < numConstants; ++constantIndex) {
        const auto &binding = executable.constantBindings.at(constantIndex);
        const auto workloadConstantIndex = binding.constantIndex;
        const auto &constant = workloadState.constants.at(workloadConstantIndex);
        const auto &resource = workloadState.resources.at(constant.resourceIndex);
        if (detail::tensorMetadata(resource).scaledTensorConfig) {
#ifdef VK_ARM_tensors2
            appendScaledGraphConstant(storage, binding.graphConstantId, constant, resource);
            continue;
#else
            throw std::runtime_error("Scaled graph constants require VK_ARM_tensors2 headers");
#endif
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
