/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <vulkan/vulkan_beta.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlsdk::workloadlib::detail {

/*******************************************************************************
 * Workload metadata
 *******************************************************************************/

ResourceKind resourceKind(vk::DescriptorType descriptorType) noexcept {
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

std::string_view resourceKindName(ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::Tensor:
        return "tensor";
    case ResourceKind::StorageBuffer:
        return "storage buffer";
    case ResourceKind::Image:
        return "image";
    case ResourceKind::Unknown:
        return "unknown";
    }
    return "unknown";
}

vk::DeviceSize elementCount(const std::vector<int64_t> &shape) noexcept {
    if (shape.empty()) {
        return 0;
    }

    vk::DeviceSize count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) {
            return 0;
        }
        count *= static_cast<vk::DeviceSize>(dimension);
    }
    return count;
}

vk::DeviceSize storageBufferByteSize(vk::DeviceSize explicitByteSize, vk::Format format,
                                     const std::vector<int64_t> &shape, const std::vector<int64_t> &stride) {
    if (explicitByteSize != 0) {
        return explicitByteSize;
    }
    if (!stride.empty() && stride.size() != shape.size()) {
        throw std::runtime_error("Storage buffer stride rank must match shape rank");
    }

    vk::DeviceSize elementSize = 0;
    switch (format) {
    case vk::Format::eR8Sint:
        elementSize = 1;
        break;
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
        elementSize = 4;
        break;
    default:
        throw std::runtime_error("Session does not support storage buffer format " +
                                 std::to_string(static_cast<uint32_t>(format)));
    }

    if (!stride.empty()) {
        vk::DeviceSize size = elementSize;
        for (uint32_t i = 0; i < shape.size(); ++i) {
            if (shape[i] <= 0 || stride[i] < 0) {
                return 0;
            }
            size += static_cast<vk::DeviceSize>(shape[i] - 1) * static_cast<vk::DeviceSize>(stride[i]);
        }
        return size;
    }

    return elementCount(shape) * elementSize;
}

void validateSpecializationInfo(const SpecializationInfo &specializationInfo, std::string_view description) {
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

/*******************************************************************************
 * Extension support
 *******************************************************************************/

bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, std::string_view name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
        return std::string_view(extension.extensionName.data()) == name;
    });
}

bool hasExtension(const std::vector<const char *> &extensions, std::string_view name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto *extension) {
        return extension != nullptr && std::string_view(extension) == name;
    });
}

bool hasExtensions(const std::vector<vk::ExtensionProperties> &availableExtensions,
                   const std::vector<const char *> &requiredExtensions) {
    return std::all_of(requiredExtensions.begin(), requiredExtensions.end(), [&availableExtensions](const auto *name) {
        return name != nullptr && hasExtension(availableExtensions, name);
    });
}

void appendExtension(std::vector<const char *> &extensions, const char *name) {
    if (name == nullptr) {
        throw std::runtime_error("Vulkan extension name must not be null");
    }
    if (!hasExtension(extensions, name)) {
        extensions.push_back(name);
    }
}

std::vector<const char *> requiredWorkloadDeviceExtensions(const std::vector<const char *> &additionalExtensions) {
    std::vector<const char *> extensions = {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                            VK_KHR_MAINTENANCE_5_EXTENSION_NAME, VK_ARM_DATA_GRAPH_EXTENSION_NAME,
                                            VK_ARM_TENSORS_EXTENSION_NAME};
    for (const auto *extension : additionalExtensions) {
        appendExtension(extensions, extension);
    }
    return extensions;
}

/*******************************************************************************
 * Device configuration
 *******************************************************************************/

RuntimeDeviceConfiguration::RuntimeDeviceConfiguration(const std::vector<vk::ExtensionProperties> &availableExtensions,
                                                       std::vector<const char *> deviceExtensions,
                                                       void *deviceFeaturePNext)
    : deviceExtensions_(std::move(deviceExtensions)), featureChain_(&dataGraphFeatures_) {
    deviceFeatures_.shaderInt16 = true;
    deviceFeatures_.shaderInt64 = true;

    vulkan12Features_.storageBuffer8BitAccess = true;
    vulkan12Features_.shaderInt8 = true;
    vulkan12Features_.shaderFloat16 = true;
    vulkan12Features_.vulkanMemoryModel = true;
    vulkan12Features_.pNext = deviceFeaturePNext;

    vulkan13Features_.synchronization2 = true;
    vulkan13Features_.pipelineCreationCacheControl = true;
    vulkan13Features_.pNext = &vulkan12Features_;

    maintenance5Features_.maintenance5 = true;
    maintenance5Features_.pNext = &vulkan13Features_;

    tensorFeatures_.tensors = true;
    tensorFeatures_.shaderTensorAccess = true;
    tensorFeatures_.tensorNonPacked = true;
    tensorFeatures_.pNext = &maintenance5Features_;

    dataGraphFeatures_.dataGraph = true;
    dataGraphFeatures_.dataGraphShaderModule = true;
    dataGraphFeatures_.dataGraphSpecializationConstants = true;
    dataGraphFeatures_.pNext = &tensorFeatures_;

    if (hasExtension(availableExtensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) {
        replicatedCompositesFeatures_.shaderReplicatedComposites = true;
        replicatedCompositesFeatures_.pNext = featureChain_;
        featureChain_ = &replicatedCompositesFeatures_;
        appendExtension(deviceExtensions_, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
    }
    // Enable extension if available, as it is required for some platforms (e.g. MoltenVK on Darwin).
    if (hasExtension(availableExtensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        appendExtension(deviceExtensions_, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
}

const std::vector<const char *> &RuntimeDeviceConfiguration::extensions() const noexcept { return deviceExtensions_; }

const vk::PhysicalDeviceFeatures &RuntimeDeviceConfiguration::features() const noexcept { return deviceFeatures_; }

const void *RuntimeDeviceConfiguration::featureChain() const noexcept { return featureChain_; }

/*******************************************************************************
 * Device resources
 *******************************************************************************/

std::optional<uint32_t> findQueueFamilyIndex(const vk::raii::PhysicalDevice &physicalDevice,
                                             vk::QueueFlags requiredFlags) {
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i) {
        if ((queueFamilies[i].queueFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    return std::nullopt;
}

uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t memoryTypeBits,
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
 * Executable synchronization
 *******************************************************************************/

vk::PipelineBindPoint pipelineBindPoint(ExecutableKind executableKind) {
    switch (executableKind) {
    case ExecutableKind::Graph:
        return vk::PipelineBindPoint::eDataGraphARM;
    case ExecutableKind::Compute:
        return vk::PipelineBindPoint::eCompute;
    }
    throw std::runtime_error("Unsupported executable kind for pipeline bind point");
}

vk::PipelineStageFlags2 pipelineStage(ExecutableKind executableKind) {
    switch (executableKind) {
    case ExecutableKind::Graph:
        return vk::PipelineStageFlagBits2::eDataGraphARM;
    case ExecutableKind::Compute:
        return vk::PipelineStageFlagBits2::eComputeShader;
    }
    throw std::runtime_error("Unsupported executable kind for pipeline stage");
}

vk::AccessFlags2 readAccess(ExecutableKind executableKind) {
    switch (executableKind) {
    case ExecutableKind::Graph:
        return vk::AccessFlagBits2::eDataGraphReadARM;
    case ExecutableKind::Compute:
        return vk::AccessFlagBits2::eShaderRead;
    }
    throw std::runtime_error("Unsupported executable kind for read access");
}

vk::AccessFlags2 writeAccess(ExecutableKind executableKind) {
    switch (executableKind) {
    case ExecutableKind::Graph:
        return vk::AccessFlagBits2::eDataGraphWriteARM;
    case ExecutableKind::Compute:
        return vk::AccessFlagBits2::eShaderWrite;
    }
    throw std::runtime_error("Unsupported executable kind for write access");
}

/*******************************************************************************
 * Image metadata
 *******************************************************************************/

vk::ImageLayout imageLayout(vk::DescriptorType descriptorType) {
    switch (descriptorType) {
    case vk::DescriptorType::eCombinedImageSampler:
        return vk::ImageLayout::eShaderReadOnlyOptimal;
    case vk::DescriptorType::eStorageImage:
        return vk::ImageLayout::eGeneral;
    default:
        throw std::runtime_error("Descriptor type does not have an image layout");
    }
}

vk::ImageUsageFlags imageUsage(vk::DescriptorType descriptorType, bool aliased) {
    vk::ImageUsageFlags usage;
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

vk::AccessFlags2 imageAccess(ExecutableKind executableKind, vk::DescriptorType descriptorType) {
    if (descriptorType == vk::DescriptorType::eCombinedImageSampler) {
        return readAccess(executableKind);
    }
    if (descriptorType == vk::DescriptorType::eStorageImage) {
        return readAccess(executableKind) | writeAccess(executableKind);
    }
    throw std::runtime_error("Descriptor type does not have image access flags");
}

/*******************************************************************************
 * Descriptor sets
 *******************************************************************************/

std::vector<vk::DescriptorSetLayout>
rawDescriptorSetLayouts(const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts) {
    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(descriptorSetLayouts.size());
    for (const auto &layout : descriptorSetLayouts) {
        layouts.push_back(*layout);
    }
    return layouts;
}

} // namespace mlsdk::workloadlib::detail
