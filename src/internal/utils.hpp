/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "workload_impl.hpp"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlworkloadlib::detail {

/*******************************************************************************
 * General utilities
 *******************************************************************************/

namespace utils {

/*******************************************************************************
 * Error and string helpers
 *******************************************************************************/

[[noreturn]] inline void throwNotImplemented(const char *api) {
    throw std::runtime_error(std::string(api) + " is not implemented yet");
}

inline std::string_view resourceKindName(ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::Unknown:
        return "Unknown";
    case ResourceKind::Tensor:
        return "Tensor";
    case ResourceKind::StorageBuffer:
        return "StorageBuffer";
    case ResourceKind::Image:
        return "Image";
    }
    return "Unhandled";
}

/*******************************************************************************
 * Container helpers
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

/*******************************************************************************
 * Shape and sparsity helpers
 *******************************************************************************/

inline vk::DeviceSize elementCount(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
        return 0;
    }
    vk::DeviceSize elements = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0) {
            return 0;
        }
        elements *= static_cast<vk::DeviceSize>(dimension);
    }
    return elements;
}

constexpr bool isSparsityDimensionSpecified(int64_t dimension) noexcept { return dimension >= 0; }

constexpr bool isSparsityDimensionValid(int64_t dimension) noexcept {
    constexpr int64_t unspecifiedSparsityDimension = -1;
    return dimension >= unspecifiedSparsityDimension;
}

/*******************************************************************************
 * Workload metadata helpers
 *******************************************************************************/

inline uint32_t requiredPushConstantSize(const Workload &workload) {
    const auto &executables = workloadImpl(workload).executables;
    const auto firstPushConstantExecutable =
        std::find_if(executables.begin(), executables.end(),
                     [](const auto &executable) { return executable.pushConstantSize != 0; });
    if (firstPushConstantExecutable == executables.end()) {
        return 0;
    }

    const auto size = firstPushConstantExecutable->pushConstantSize;
    const auto hasDifferentPushConstantSize =
        std::any_of(std::next(firstPushConstantExecutable), executables.end(), [size](const auto &executable) {
            return executable.pushConstantSize != 0 && executable.pushConstantSize != size;
        });
    if (hasDifferentPushConstantSize) {
        throw std::runtime_error("Workloads with multiple push constant sizes are not supported");
    }
    return size;
}

inline void validateSpecializationInfo(const SpecializationInfo &specializationInfo, std::string_view description) {
    const auto context = std::string(description) + " specialization info";
    if (specializationInfo.mapEntries.empty()) {
        if (!specializationInfo.data.empty()) {
            throw std::runtime_error(context + " has data without map entries");
        }
        return;
    }
    if (specializationInfo.data.empty()) {
        throw std::runtime_error(context + " data must not be empty");
    }

    std::vector<uint32_t> constantIds;
    constantIds.reserve(specializationInfo.mapEntries.size());
    for (const auto &mapEntry : specializationInfo.mapEntries) {
        if (mapEntry.size == 0) {
            throw std::runtime_error(context + " map entry size must not be zero");
        }
        const auto offset = static_cast<std::size_t>(mapEntry.offset);
        if (offset > specializationInfo.data.size() || mapEntry.size > specializationInfo.data.size() - offset) {
            throw std::runtime_error(context + " map entry exceeds data size");
        }
        if (std::find(constantIds.begin(), constantIds.end(), mapEntry.constantID) != constantIds.end()) {
            throw std::runtime_error(context + " constant ids must be unique");
        }
        constantIds.push_back(mapEntry.constantID);
    }
}

inline ResourceAccess accessForRole(Resource::Role role) {
    switch (role) {
    case Resource::Role::Input:
        return ResourceAccess::Read;
    case Resource::Role::Output:
        return ResourceAccess::Write;
    case Resource::Role::Constant:
        return ResourceAccess::Read;
    case Resource::Role::Intermediate:
        return ResourceAccess::ReadWrite;
    }
    throw std::runtime_error("Unsupported workload resource role");
}

inline ResourceAccess resourceAccess(const Workload &workload, uint32_t resourceIndex) {
    const auto &workloadState = workloadImpl(workload);
    const auto &resource = workloadState.resources.at(resourceIndex);
    std::optional<ResourceAccess> access;
    for (const auto &executable : workloadState.executables) {
        for (const auto &descBinding : executable.bindings) {
            if (descBinding.resourceIndex != resourceIndex) {
                continue;
            }
            if (!access.has_value()) {
                access = descBinding.access;
            } else if (*access != descBinding.access) {
                // A public resource used through both read and write bindings must be exposed as read-write.
                return ResourceAccess::ReadWrite;
            }
        }
    }
    return access.value_or(accessForRole(resource.role));
}

inline vk::DescriptorType descriptorType(const Workload &workload, const DescriptorBinding &descBinding) {
    const auto &workloadState = workloadImpl(workload);
    const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
    if (!resource.descriptorType.has_value()) {
        throw std::runtime_error("Descriptor binding references a workload resource without descriptor type");
    }
    return *resource.descriptorType;
}

} // namespace utils

/*******************************************************************************
 * Vulkan helpers
 *******************************************************************************/

namespace vulkan_helpers {

/*******************************************************************************
 * Resource metadata helpers
 *******************************************************************************/

inline ResourceKind resourceKind(vk::DescriptorType descriptorType) {
    switch (descriptorType) {
    case vk::DescriptorType::eTensorARM:
        return ResourceKind::Tensor;
    case vk::DescriptorType::eStorageBuffer:
        return ResourceKind::StorageBuffer;
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage:
        return ResourceKind::Image;
    default:
        return ResourceKind::Unknown;
    }
}

inline vk::DeviceSize formatByteSize(vk::Format format) {
    switch (format) {
    case vk::Format::eR8Sint:
        return 1;
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
        return 4;
    default:
        throw std::runtime_error("Session does not support storage buffer format " +
                                 std::to_string(static_cast<uint32_t>(format)));
    }
}

inline vk::DeviceSize resourceByteSize(const Resource &resource) {
    const auto &metadata = bufferMetadata(resource);
    if (metadata.byteSize != 0) {
        return metadata.byteSize;
    }

    const auto elementSize = formatByteSize(resource.format);

    if (!resource.stride.empty()) {
        vk::DeviceSize size = elementSize;
        for (uint32_t i = 0; i < resource.shape.size(); ++i) {
            size +=
                static_cast<vk::DeviceSize>(resource.shape[i] - 1) * static_cast<vk::DeviceSize>(resource.stride[i]);
        }
        return size;
    }

    return utils::elementCount(resource.shape) * elementSize;
}

/*******************************************************************************
 * Executable synchronization helpers
 *******************************************************************************/

inline vk::PipelineBindPoint bindPoint(ExecutableKind type) {
    switch (type) {
    case ExecutableKind::Graph:
        return vk::PipelineBindPoint::eDataGraphARM;
    case ExecutableKind::Compute:
        return vk::PipelineBindPoint::eCompute;
    }
    throw std::runtime_error("Unsupported executable kind for pipeline bind point");
}

inline vk::PipelineStageFlags2 pipelineStage(ExecutableKind type) {
    switch (type) {
    case ExecutableKind::Graph:
        return vk::PipelineStageFlagBits2::eDataGraphARM;
    case ExecutableKind::Compute:
        return vk::PipelineStageFlagBits2::eComputeShader;
    }
    throw std::runtime_error("Unsupported executable kind for pipeline stage");
}

inline vk::AccessFlags2 readAccess(ExecutableKind type) {
    switch (type) {
    case ExecutableKind::Graph:
        return vk::AccessFlagBits2::eDataGraphReadARM;
    case ExecutableKind::Compute:
        return vk::AccessFlagBits2::eShaderRead;
    }
    throw std::runtime_error("Unsupported executable kind for read access");
}

inline vk::AccessFlags2 writeAccess(ExecutableKind type) {
    switch (type) {
    case ExecutableKind::Graph:
        return vk::AccessFlagBits2::eDataGraphWriteARM;
    case ExecutableKind::Compute:
        return vk::AccessFlagBits2::eShaderWrite;
    }
    throw std::runtime_error("Unsupported executable kind for write access");
}

/*******************************************************************************
 * Image metadata helpers
 *******************************************************************************/

inline vk::Extent3D imageExtentFromMetadata(const Resource &resource) {
    const auto &metadata = imageMetadata(resource);
    if (metadata.extent.width != 0 || metadata.extent.height != 0 || metadata.extent.depth != 0) {
        return metadata.extent;
    }
    const auto dimension = [](std::string_view name, int64_t value) {
        constexpr auto maxDimension = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        if (value <= 0 || value > maxDimension) {
            throw std::runtime_error("Workload image " + std::string(name) + " must be between 1 and UINT32_MAX; got " +
                                     std::to_string(value));
        }
        return static_cast<uint32_t>(value);
    };

    if (resource.shape.size() == 3) {
        return {dimension("width", resource.shape[0]), dimension("height", resource.shape[1]),
                dimension("depth", resource.shape[2])};
    }
    if (resource.shape.size() != 4) {
        throw std::runtime_error("Workload image shape must be a 3D extent or 4D NHWC; got rank " +
                                 std::to_string(resource.shape.size()));
    }
    if (resource.shape[0] != 1) {
        throw std::runtime_error("Workload NHWC image batch must be 1; got " + std::to_string(resource.shape[0]));
    }
    if (resource.shape[3] != 4) {
        throw std::runtime_error("Workload NHWC image channel count must be 4; got " +
                                 std::to_string(resource.shape[3]));
    }
    return {dimension("width", resource.shape[2]), dimension("height", resource.shape[1]), 1};
}

inline void validateImageFormat(vk::Format format) {
    if (format != vk::Format::eR8G8B8A8Snorm) {
        throw std::runtime_error("Session only supports eR8G8B8A8Snorm image resources");
    }
}

inline vk::ImageLayout imageLayout(vk::DescriptorType descriptorType) {
    switch (descriptorType) {
    case vk::DescriptorType::eCombinedImageSampler:
        return vk::ImageLayout::eShaderReadOnlyOptimal;
    case vk::DescriptorType::eStorageImage:
        return vk::ImageLayout::eGeneral;
    default:
        throw std::runtime_error("Descriptor type does not have an image layout");
    }
}

inline vk::ImageUsageFlags imageUsage(vk::DescriptorType descriptorType, bool aliased) {
    vk::ImageUsageFlags usage{};
    switch (descriptorType) {
    case vk::DescriptorType::eCombinedImageSampler:
        usage |= vk::ImageUsageFlagBits::eSampled;
        break;
    case vk::DescriptorType::eStorageImage:
        usage |= vk::ImageUsageFlagBits::eStorage;
        break;
    default:
        throw std::runtime_error("Descriptor type does not have image usage flags");
    }
    if (aliased) {
        usage |= vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                 vk::ImageUsageFlagBits::eTensorAliasingARM;
    }
    return usage;
}

inline vk::AccessFlags2 imageAccess(ExecutableKind type, vk::DescriptorType descriptorType) {
    if (descriptorType == vk::DescriptorType::eCombinedImageSampler) {
        return readAccess(type);
    }
    if (descriptorType == vk::DescriptorType::eStorageImage) {
        return readAccess(type) | writeAccess(type);
    }
    throw std::runtime_error("Descriptor type does not have image access flags");
}

/*******************************************************************************
 * Memory helpers
 *******************************************************************************/

inline uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t memoryTypeBits,
                               vk::MemoryPropertyFlags requiredFlags) {
    const auto memoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool supportsType = (memoryTypeBits & (uint32_t{1} << i)) != 0;
        const bool hasFlags = (memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags;
        if (supportsType && hasFlags) {
            return i;
        }
    }
    throw std::runtime_error("Cannot find a compatible memory type");
}

/*******************************************************************************
 * Descriptor set helpers
 *******************************************************************************/

inline std::vector<std::vector<DescriptorBinding>>
splitBindingsBySet(const std::vector<DescriptorBinding> &descBindings) {
    std::vector<std::vector<DescriptorBinding>> sets;
    for (const auto &descBinding : descBindings) {
        while (sets.size() <= descBinding.set) {
            sets.emplace_back();
        }
        sets[descBinding.set].push_back(descBinding);
    }
    return sets;
}

inline std::vector<vk::DescriptorSetLayout>
rawLayouts(const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts) {
    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(descriptorSetLayouts.size());
    for (const auto &layout : descriptorSetLayouts) {
        layouts.push_back(*layout);
    }
    return layouts;
}

} // namespace vulkan_helpers
} // namespace mlworkloadlib::detail
