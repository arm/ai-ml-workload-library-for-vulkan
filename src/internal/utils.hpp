/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/workload_types.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace mlsdk::workloadlib::detail {

/*******************************************************************************
 * Workload metadata
 *******************************************************************************/

ResourceKind resourceKind(vk::DescriptorType descriptorType) noexcept;
std::string_view resourceKindName(ResourceKind kind) noexcept;

vk::DeviceSize elementCount(const std::vector<int64_t> &shape) noexcept;
vk::DeviceSize storageBufferByteSize(vk::DeviceSize explicitByteSize, vk::Format format,
                                     const std::vector<int64_t> &shape, const std::vector<int64_t> &stride);

void validateSpecializationInfo(const SpecializationInfo &specializationInfo, std::string_view description);

/*******************************************************************************
 * Extension support
 *******************************************************************************/

bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, std::string_view name);
bool hasExtension(const std::vector<const char *> &extensions, std::string_view name);
bool hasExtensions(const std::vector<vk::ExtensionProperties> &availableExtensions,
                   const std::vector<const char *> &requiredExtensions);
void appendExtension(std::vector<const char *> &extensions, const char *name);
std::vector<const char *> requiredWorkloadDeviceExtensions(const std::vector<const char *> &additionalExtensions = {});

/*******************************************************************************
 * Device configuration
 *******************************************************************************/

class RuntimeDeviceConfiguration {
  public:
    RuntimeDeviceConfiguration(const std::vector<vk::ExtensionProperties> &availableExtensions,
                               std::vector<const char *> deviceExtensions, void *deviceFeaturePNext = nullptr);
    RuntimeDeviceConfiguration(const RuntimeDeviceConfiguration &) = delete;
    RuntimeDeviceConfiguration &operator=(const RuntimeDeviceConfiguration &) = delete;
    RuntimeDeviceConfiguration(RuntimeDeviceConfiguration &&) = delete;
    RuntimeDeviceConfiguration &operator=(RuntimeDeviceConfiguration &&) = delete;
    ~RuntimeDeviceConfiguration() = default;

    const std::vector<const char *> &extensions() const noexcept;
    const vk::PhysicalDeviceFeatures &features() const noexcept;
    const void *featureChain() const noexcept;

  private:
    vk::PhysicalDeviceFeatures deviceFeatures_;
    vk::PhysicalDeviceVulkan12Features vulkan12Features_;
    vk::PhysicalDeviceVulkan13Features vulkan13Features_;
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5Features_;
    vk::PhysicalDeviceTensorFeaturesARM tensorFeatures_;
    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeatures_;
    vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures_;
    std::vector<const char *> deviceExtensions_;
    void *featureChain_;
};

/*******************************************************************************
 * Device resources
 *******************************************************************************/

std::optional<uint32_t> findQueueFamilyIndex(const vk::raii::PhysicalDevice &physicalDevice,
                                             vk::QueueFlags requiredFlags);
uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t memoryTypeBits,
                        vk::MemoryPropertyFlags requiredFlags);

/*******************************************************************************
 * Executable synchronization
 *******************************************************************************/

vk::PipelineBindPoint pipelineBindPoint(ExecutableKind executableKind);
vk::PipelineStageFlags2 pipelineStage(ExecutableKind executableKind);
vk::AccessFlags2 readAccess(ExecutableKind executableKind);
vk::AccessFlags2 writeAccess(ExecutableKind executableKind);

/*******************************************************************************
 * Image metadata
 *******************************************************************************/

vk::ImageLayout imageLayout(vk::DescriptorType descriptorType);
vk::ImageUsageFlags imageUsage(vk::DescriptorType descriptorType, bool aliased);
vk::AccessFlags2 imageAccess(ExecutableKind executableKind, vk::DescriptorType descriptorType);

/*******************************************************************************
 * Descriptor sets
 *******************************************************************************/

std::vector<vk::DescriptorSetLayout>
rawDescriptorSetLayouts(const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts);

} // namespace mlsdk::workloadlib::detail
