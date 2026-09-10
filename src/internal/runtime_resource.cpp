/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime_resource.hpp"
#include "utils.hpp"

#include <vulkan/vulkan_format_traits.hpp>

#include <limits>
#include <sstream>
#include <string_view>

namespace mlsdk::workloadlib::detail {

void validateAliasTensorDescription(uint32_t resourceIndex, const Resource &resource,
                                    const std::vector<int64_t> &strides,
                                    const vk::PhysicalDeviceTensorPropertiesARM *limits) {
    const auto fail = [resourceIndex](const auto... details) {
        std::ostringstream message;
        message << "Alias tensor resource " << resourceIndex << ": ";
        (message << ... << details);
        throw std::runtime_error(message.str());
    };
    const auto rank = resource.shape.size();
    if (rank == 0 || (!strides.empty() && strides.size() != rank)) {
        fail("stride count ", strides.size(), " must match the non-empty tensor rank ", rank);
    }
    if (limits && rank > limits->maxTensorDimensionCount) {
        fail("rank ", rank, " exceeds maxTensorDimensionCount ", limits->maxTensorDimensionCount);
    }
    const uint32_t elementSize = vk::blockSize(resource.format);
    if (vk::componentCount(resource.format) != 1 || elementSize == 0 || vk::isCompressed(resource.format)) {
        const auto formatName = vk::to_string(resource.format);
        fail("format ", std::string_view(formatName), " must be scalar and uncompressed");
    }
    const auto maximumSize = limits ? limits->maxTensorSize : uint64_t{std::numeric_limits<int64_t>::max()};
    uint64_t elements = 1;
    uint64_t innerExtent = elementSize;
    for (size_t index = rank; index-- > 0;) {
        const auto dimension = resource.shape[index];
        if (dimension <= 0) {
            fail("dimension ", index, " size ", dimension, " must be positive");
        }
        if (limits && static_cast<uint64_t>(dimension) > limits->maxPerDimensionTensorElements) {
            fail("dimension ", index, " size ", dimension, " exceeds maxPerDimensionTensorElements ",
                 limits->maxPerDimensionTensorElements);
        }
        if (limits && elements > limits->maxTensorElements / static_cast<uint64_t>(dimension)) {
            fail("element count exceeds maxTensorElements ", limits->maxTensorElements, " at dimension ", index, " (",
                 elements, " * ", dimension, ")");
        }
        elements *= static_cast<uint64_t>(dimension);
        uint64_t stride = innerExtent;
        if (!strides.empty()) {
            if (strides[index] <= 0 || strides[index] % elementSize != 0) {
                fail("dimension ", index, ": byte stride ", strides[index],
                     " must be positive and element-aligned (element size ", elementSize, ")");
            }
            if (limits && strides[index] > limits->maxTensorStride) {
                fail("dimension ", index, ": stride ", strides[index], " exceeds maxTensorStride ",
                     limits->maxTensorStride);
            }
            stride = static_cast<uint64_t>(strides[index]);
            if (stride < innerExtent) {
                fail("strides overlap adjacent dimensions: dimension ", index, " stride ", stride,
                     " is smaller than the minimum ", innerExtent);
            }
            if (index == rank - 1 && stride != elementSize) {
                fail("innermost stride ", stride, " must equal the element size ", elementSize);
            }
        }
        if (stride > maximumSize / static_cast<uint64_t>(dimension)) {
            fail(limits ? "byte extent exceeds maxTensorSize " : "byte extent overflows int64_t ", maximumSize,
                 " at dimension ", index, " (stride ", stride, ", size ", dimension, ")");
        }
        innerExtent = stride * static_cast<uint64_t>(dimension);
    }
}

// Returned descriptions borrow immutable Workload and Session storage during the Vulkan call.
vk::TensorDescriptionARM tensorDescription(const Workload &workload, uint32_t resourceIndex, vk::TensorTilingARM tiling,
                                           const std::vector<int64_t> &strides) {
    const auto &resource = workloadImpl(workload).resources.at(resourceIndex);
    return {tiling,
            resource.format,
            static_cast<uint32_t>(resource.shape.size()),
            resource.shape.data(),
            strides.empty() ? nullptr : strides.data(),
            tensorUsageForTiling(workload, resourceIndex, tiling)};
}

vk::raii::TensorARM createIntermediateTensor(const Workload &workload, const ContextView &contextView,
                                             uint32_t resourceIndex, vk::TensorTilingARM tiling,
                                             const std::vector<int64_t> &strides) {
    const auto description = tensorDescription(workload, resourceIndex, tiling, strides);
    validateTensorTiling(tiling, description.usage, workloadImpl(workload).resources.at(resourceIndex).shape,
                         !strides.empty());
    const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);
    return {contextView.device.get(), createInfo};
}

vk::raii::Buffer createIntermediateBuffer(const Workload &workload, const ContextView &contextView,
                                          uint32_t resourceIndex) {
    const auto &resource = workloadImpl(workload).resources.at(resourceIndex);
    const auto byteSize = detail::storageBufferByteSize(detail::bufferMetadata(resource).byteSize, resource.format,
                                                        resource.shape, resource.stride);
    const vk::BufferCreateInfo createInfo({}, byteSize, vk::BufferUsageFlagBits::eStorageBuffer);
    return {contextView.device.get(), createInfo};
}

vk::raii::Image createIntermediateImage(const Workload &workload, const ContextView &contextView,
                                        uint32_t resourceIndex, bool forAliasing, vk::ImageTiling tiling) {
    const auto &workloadState = workloadImpl(workload);
    const auto &resource = workloadState.resources.at(resourceIndex);
    const auto descriptorType = resource.descriptorType.value();

    const auto requiredUsage = detail::imageMetadata(resource).usage | detail::imageUsage(descriptorType, forAliasing);
    const auto effectiveUsage = detail::imageUsageForTiling(requiredUsage, forAliasing, tiling);

    const auto extent = resource.imageExtent();
    const auto imageType = extent.depth > 1 ? vk::ImageType::e3D : vk::ImageType::e2D;
    const vk::ImageCreateInfo createInfo(forAliasing ? vk::ImageCreateFlagBits::eAlias : vk::ImageCreateFlags{},
                                         imageType, resource.format, extent, 1, 1, vk::SampleCountFlagBits::e1, tiling,
                                         effectiveUsage, vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined);

    return {contextView.device.get(), createInfo};
}

AliasPeerMemoryRequirements queryAliasPeerMemoryRequirements(const Workload &workload, const ContextView &contextView,
                                                             uint32_t resourceIndex, ResourceTiling tiling,
                                                             const std::vector<int64_t> &strides) {
    const auto query = [&contextView](const auto &object) {
        const auto chain = aliasMemoryRequirements(contextView, *object);
        const auto &dedicated = chain.template get<vk::MemoryDedicatedRequirements>();
        return AliasPeerMemoryRequirements{chain.template get<vk::MemoryRequirements2>().memoryRequirements,
                                           dedicated.prefersDedicatedAllocation != 0,
                                           dedicated.requiresDedicatedAllocation != 0};
    };
    switch (workloadImpl(workload).resources.at(resourceIndex).descriptorType.value()) {
    case vk::DescriptorType::eTensorARM:
        return query(createIntermediateTensor(workload, contextView, resourceIndex, toTensorTiling(tiling), strides));
    case vk::DescriptorType::eStorageBuffer:
        return query(createIntermediateBuffer(workload, contextView, resourceIndex));
    case vk::DescriptorType::eStorageImage:
    case vk::DescriptorType::eCombinedImageSampler:
        return query(createIntermediateImage(workload, contextView, resourceIndex, true, toImageTiling(tiling)));
    default:
        throw std::runtime_error("Unsupported alias peer descriptor type");
    }
}

} // namespace mlsdk::workloadlib::detail
