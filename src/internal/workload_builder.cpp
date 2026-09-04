/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "workload_builder.hpp"

#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlworkloadlib::detail {

namespace {

/*******************************************************************************
 * Internal helpers
 *******************************************************************************/

vk::DescriptorType descriptorTypeForResource(const ResourceRequirements &requirements) {
    const auto descriptorKind = vulkan_helpers::resourceKind(requirements.descriptorType);
    if (descriptorKind != ResourceKind::Unknown) {
        if (requirements.kind != ResourceKind::Unknown && requirements.kind != descriptorKind) {
            throw std::runtime_error(
                "Workload resource kind " + std::string(utils::resourceKindName(requirements.kind)) +
                " does not match descriptor kind " + std::string(utils::resourceKindName(descriptorKind)));
        }
        return requirements.descriptorType;
    }

    switch (requirements.kind) {
    case ResourceKind::Tensor:
        return vk::DescriptorType::eTensorARM;
    case ResourceKind::StorageBuffer:
        return vk::DescriptorType::eStorageBuffer;
    case ResourceKind::Image:
        throw std::runtime_error("Image resources must specify sampled or storage image descriptor type");
    case ResourceKind::Unknown:
        throw std::runtime_error("Workload resource kind is not specified");
    }
    throw std::logic_error("Unhandled workload resource kind");
}

ResourceKind kindForResource(const ResourceRequirements &requirements, vk::DescriptorType descriptorType) {
    if (requirements.kind != ResourceKind::Unknown) {
        return requirements.kind;
    }

    const auto kind = vulkan_helpers::resourceKind(descriptorType);
    if (kind == ResourceKind::Unknown) {
        throw std::runtime_error("Workload descriptor type is not supported");
    }
    return kind;
}

Resource resourceBase(std::string name, Resource::Role role, std::optional<vk::DescriptorType> descriptorType,
                      vk::Format format, std::vector<int64_t> shape, std::vector<int64_t> stride,
                      vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId, bool requiresBoundMemoryInfo) {
    if (!stride.empty() && stride.size() != shape.size()) {
        throw std::runtime_error("Workload resource stride rank must undefined or match shape rank");
    }

    Resource workloadResource;
    workloadResource.name = std::move(name);
    workloadResource.role = role;
    workloadResource.descriptorType = descriptorType;
    workloadResource.format = format;
    workloadResource.shape = std::move(shape);
    workloadResource.stride = std::move(stride);
    workloadResource.elementCount = elementCount;
    workloadResource.aliasGroupId = aliasGroupId;
    workloadResource.requiresBoundMemoryInfo = requiresBoundMemoryInfo;
    return workloadResource;
}

Resource::ImageMetadata::SamplerConfig samplerConfigForRequirements(const SamplerRequirements &requirements) {
    Resource::ImageMetadata::SamplerConfig samplerConfig;
    samplerConfig.minFilter = requirements.minFilter;
    samplerConfig.magFilter = requirements.magFilter;
    samplerConfig.mipmapMode = requirements.mipmapMode;
    samplerConfig.addressModeU = requirements.addressModeU;
    samplerConfig.addressModeV = requirements.addressModeV;
    samplerConfig.addressModeW = requirements.addressModeW;
    return samplerConfig;
}

bool isPublicRole(Resource::Role role) { return role == Resource::Role::Input || role == Resource::Role::Output; }

} // namespace

/*******************************************************************************
 * Module metadata
 *******************************************************************************/

Module moduleFromImplementation(ModuleImplementation implementation, std::string name, std::string entryPoint) {
    switch (implementation.codeKind) {
    case ModuleCodeKind::Spirv:
        if (implementation.spirv.empty()) {
            throw std::runtime_error("Programmatic workload module '" + name + "' SPIR-V code must not be empty");
        }
        break;
    case ModuleCodeKind::Glsl:
    case ModuleCodeKind::Hlsl:
        if (implementation.source.empty()) {
            throw std::runtime_error("Programmatic workload module '" + name + "' source code must not be empty");
        }
        break;
    case ModuleCodeKind::Missing:
        break;
    }

    Module workloadModule;
    workloadModule.name = std::move(name);
    workloadModule.entryPoint = std::move(entryPoint);
    workloadModule.codeKind = implementation.codeKind;
    workloadModule.code = std::move(implementation.spirv);
    workloadModule.source = std::move(implementation.source);
    workloadModule.buildOptions = std::move(implementation.buildOptions);
    workloadModule.includeDirs = std::move(implementation.includeDirs);
    return workloadModule;
}

/*******************************************************************************
 * Lifetime
 *******************************************************************************/

WorkloadBuilder::WorkloadBuilder() : workloadState_(std::make_unique<Workload::Impl>()) {}

WorkloadBuilder::WorkloadBuilder(std::unique_ptr<MemoryMap> mappedFile) : WorkloadBuilder() {
    workloadState_->mappedFile = std::move(mappedFile);
}

/*******************************************************************************
 * Resources
 *******************************************************************************/

void WorkloadBuilder::reserveResources(std::size_t count) { workloadState_->resources.reserve(count); }

uint32_t WorkloadBuilder::addResource(std::string name, const ResourceRequirements &requirements, Resource::Role role,
                                      std::optional<uint32_t> aliasGroupId) {
    const auto descriptorType = descriptorTypeForResource(requirements);
    const auto kind = kindForResource(requirements, descriptorType);
    const bool requiresBoundMemoryInfo = requirements.requiresBoundMemoryInfo || aliasGroupId.has_value();

    switch (kind) {
    case ResourceKind::Tensor:
        return addTensorResource(
            std::move(name), role, descriptorType, requirements.format, requirements.tensor.shape,
            requirements.tensor.stride, requirements.elementCount, aliasGroupId, requiresBoundMemoryInfo,
            requirements.tensor.usage ? requirements.tensor.usage : vk::TensorUsageFlagBitsARM::eDataGraph);
    case ResourceKind::StorageBuffer: {
        const auto usage =
            requirements.buffer.usage ? requirements.buffer.usage : vk::BufferUsageFlagBits::eStorageBuffer;
        return addBufferResource(std::move(name), role, descriptorType, requirements.format, {}, {},
                                 requirements.elementCount, aliasGroupId, requiresBoundMemoryInfo,
                                 requirements.buffer.byteSize, usage);
    }
    case ResourceKind::Image: {
        const auto usage =
            requirements.image.usage ? requirements.image.usage : vulkan_helpers::imageUsage(descriptorType, false);
        std::optional<Resource::ImageMetadata::SamplerConfig> samplerConfig;
        if (requirements.image.runtimeSampler.has_value()) {
            samplerConfig = samplerConfigForRequirements(*requirements.image.runtimeSampler);
        }
        return addImageResource(std::move(name), role, descriptorType, requirements.format, {}, {},
                                requirements.elementCount, aliasGroupId, requiresBoundMemoryInfo,
                                requirements.image.extent, usage, requirements.image.requiredLayout,
                                requirements.image.range, samplerConfig);
    }
    case ResourceKind::Unknown:
        throw std::runtime_error("Workload resource kind is not specified");
    }
    throw std::logic_error("Unhandled workload resource kind");
}

uint32_t WorkloadBuilder::addTensorResource(std::string name, Resource::Role role,
                                            std::optional<vk::DescriptorType> descriptorType, vk::Format format,
                                            std::vector<int64_t> shape, std::vector<int64_t> stride,
                                            vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId,
                                            bool requiresBoundMemoryInfo, vk::TensorUsageFlagsARM usage) {
    auto workloadResource = resourceBase(std::move(name), role, descriptorType, format, std::move(shape),
                                         std::move(stride), elementCount, aliasGroupId, requiresBoundMemoryInfo);
    workloadResource.metadata = Resource::TensorMetadata{usage};
    return appendResource(std::move(workloadResource));
}

uint32_t WorkloadBuilder::addBufferResource(std::string name, Resource::Role role, vk::DescriptorType descriptorType,
                                            vk::Format format, std::vector<int64_t> shape, std::vector<int64_t> stride,
                                            vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId,
                                            bool requiresBoundMemoryInfo, vk::DeviceSize byteSize,
                                            vk::BufferUsageFlags usage) {
    auto workloadResource = resourceBase(std::move(name), role, descriptorType, format, std::move(shape),
                                         std::move(stride), elementCount, aliasGroupId, requiresBoundMemoryInfo);
    workloadResource.metadata = Resource::BufferMetadata{byteSize, usage};
    return appendResource(std::move(workloadResource));
}

uint32_t WorkloadBuilder::addImageResource(std::string name, Resource::Role role, vk::DescriptorType descriptorType,
                                           vk::Format format, std::vector<int64_t> shape, std::vector<int64_t> stride,
                                           vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId,
                                           bool requiresBoundMemoryInfo, vk::Extent3D extent, vk::ImageUsageFlags usage,
                                           vk::ImageLayout layout, vk::ImageSubresourceRange subresourceRange,
                                           std::optional<Resource::ImageMetadata::SamplerConfig> samplerConfig) {
    auto workloadResource = resourceBase(std::move(name), role, descriptorType, format, std::move(shape),
                                         std::move(stride), elementCount, aliasGroupId, requiresBoundMemoryInfo);
    workloadResource.metadata = Resource::ImageMetadata{extent, usage, layout, subresourceRange, samplerConfig};
    return appendResource(std::move(workloadResource));
}

Resource &WorkloadBuilder::resource(uint32_t resourceIndex) { return workloadState_->resources.at(resourceIndex); }

const Resource &WorkloadBuilder::resource(uint32_t resourceIndex) const {
    return workloadState_->resources.at(resourceIndex);
}

void WorkloadBuilder::setPublicResourceName(uint32_t resourceIndex, std::string name) {
    if (resourceIndex >= workloadState_->resources.size()) {
        return;
    }

    auto &workloadResource = workloadState_->resources.at(resourceIndex);
    if (workloadResource.role == Resource::Role::Input || workloadResource.role == Resource::Role::Output) {
        workloadResource.name = std::move(name);
    }
}

uint32_t WorkloadBuilder::appendResource(Resource workloadResource) {
    const auto resourceIndex = static_cast<uint32_t>(workloadState_->resources.size());
    workloadState_->resources.push_back(std::move(workloadResource));
    return resourceIndex;
}

/*******************************************************************************
 * Modules
 *******************************************************************************/

void WorkloadBuilder::reserveModules(std::size_t count) { workloadState_->modules.reserve(count); }

uint32_t WorkloadBuilder::addModule(std::string name, std::string entryPoint, ModuleCodeKind codeKind,
                                    std::vector<uint32_t> code, std::string source, std::string buildOptions,
                                    std::vector<std::filesystem::path> includeDirs) {
    switch (codeKind) {
    case ModuleCodeKind::Spirv:
        if (code.empty()) {
            throw std::runtime_error("Workload SPIR-V module code must not be empty");
        }
        break;
    case ModuleCodeKind::Glsl:
    case ModuleCodeKind::Hlsl:
        if (source.empty()) {
            throw std::runtime_error("Workload source module code must not be empty");
        }
        break;
    case ModuleCodeKind::Missing:
        break;
    }

    Module workloadModule;
    workloadModule.name = std::move(name);
    workloadModule.entryPoint = std::move(entryPoint);
    workloadModule.codeKind = codeKind;
    workloadModule.code = std::move(code);
    workloadModule.source = std::move(source);
    workloadModule.buildOptions = std::move(buildOptions);
    workloadModule.includeDirs = std::move(includeDirs);
    return appendModule(std::move(workloadModule));
}

uint32_t WorkloadBuilder::addModule(ModuleImplementation implementation, std::string name, std::string entryPoint) {
    return appendModule(moduleFromImplementation(std::move(implementation), std::move(name), std::move(entryPoint)));
}

uint32_t WorkloadBuilder::appendModule(Module workloadModule) {
    const auto moduleIndex = static_cast<uint32_t>(workloadState_->modules.size());
    workloadState_->modules.push_back(std::move(workloadModule));
    return moduleIndex;
}

/*******************************************************************************
 * Constants
 *******************************************************************************/

void WorkloadBuilder::reserveConstants(std::size_t count) { workloadState_->constants.reserve(count); }

uint32_t WorkloadBuilder::addConstantResource(std::string name, const ResourceRequirements &requirements,
                                              std::optional<uint32_t> aliasGroupId) {
    return addTensorResource(std::move(name), Resource::Role::Constant, std::nullopt, requirements.format,
                             requirements.tensor.shape, requirements.tensor.stride, requirements.elementCount,
                             aliasGroupId, requirements.requiresBoundMemoryInfo || aliasGroupId.has_value(),
                             requirements.tensor.usage ? requirements.tensor.usage
                                                       : vk::TensorUsageFlagBitsARM::eDataGraph);
}

uint32_t WorkloadBuilder::addConstant(uint32_t resourceIndex, const void *data, std::uint64_t size,
                                      int64_t sparsityDimension) {
    if (size != 0 && data == nullptr) {
        throw std::runtime_error("Graph constant payload is null");
    }
    if (size > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Graph constant payload is too large");
    }

    (void)resource(resourceIndex);
    Constant workloadConstant;
    workloadConstant.resourceIndex = resourceIndex;
    workloadConstant.payloadView =
        ArrayView<const uint8_t>(static_cast<const uint8_t *>(data), static_cast<std::size_t>(size));
    workloadConstant.sparsityDimension = sparsityDimension;
    return appendConstant(workloadConstant);
}

uint32_t WorkloadBuilder::appendConstant(Constant workloadConstant) {
    const auto constantIndex = static_cast<uint32_t>(workloadState_->constants.size());
    workloadState_->constants.push_back(workloadConstant);
    return constantIndex;
}

/*******************************************************************************
 * Executables
 *******************************************************************************/

void WorkloadBuilder::reserveExecutables(std::size_t count) { workloadState_->executables.reserve(count); }

uint32_t WorkloadBuilder::addExecutable(std::string name, ExecutableKind type, uint32_t moduleIndex) {
    (void)workloadState_->modules.at(moduleIndex);

    const auto executableIndex = static_cast<uint32_t>(workloadState_->executables.size());
    Executable workloadExecutable;
    workloadExecutable.executableIndex = executableIndex;
    workloadExecutable.name = std::move(name);
    workloadExecutable.type = type;
    workloadExecutable.moduleIndex = moduleIndex;
    workloadState_->executables.push_back(std::move(workloadExecutable));
    return executableIndex;
}

void WorkloadBuilder::addDescriptorBinding(uint32_t executableIndex, uint32_t resourceIndex, uint32_t set,
                                           uint32_t binding, ResourceAccess access) {
    (void)workloadState_->resources.at(resourceIndex);
    workloadState_->executables.at(executableIndex).bindings.push_back({resourceIndex, set, binding, access});
}

Executable &WorkloadBuilder::executable(uint32_t executableIndex) {
    return workloadState_->executables.at(executableIndex);
}

const Executable &WorkloadBuilder::executable(uint32_t executableIndex) const {
    return workloadState_->executables.at(executableIndex);
}

void WorkloadBuilder::setDispatchShape(uint32_t executableIndex, DispatchShape dispatch) {
    executable(executableIndex).dispatchShape = {dispatch.x, dispatch.y, dispatch.z};
}

void WorkloadBuilder::setDispatchShape(uint32_t executableIndex, std::array<uint32_t, 3> dispatch) {
    executable(executableIndex).dispatchShape = dispatch;
}

void WorkloadBuilder::setImplicitBarrier(uint32_t executableIndex, bool implicitBarrier) {
    executable(executableIndex).implicitBarrier = implicitBarrier;
}

void WorkloadBuilder::setSpecializationInfo(uint32_t executableIndex, SpecializationInfo specializationInfo) {
    workloadState_->executables.at(executableIndex).specializationInfo = std::move(specializationInfo);
}

void WorkloadBuilder::addPushConstantRange(uint32_t executableIndex, vk::ShaderStageFlags stageFlags, uint32_t offset,
                                           uint32_t size) {
    if (!stageFlags) {
        throw std::runtime_error("Push constant range stage flags must not be empty");
    }
    if (size == 0) {
        throw std::runtime_error("Push constant range size must not be zero");
    }
    if (offset % 4 != 0 || size % 4 != 0) {
        throw std::runtime_error("Push constant range offset and size must be multiples of 4");
    }
    if (offset > std::numeric_limits<uint32_t>::max() - size) {
        throw std::runtime_error("Push constant range offset and size overflow");
    }

    auto &executableState = workloadState_->executables.at(executableIndex);
    executableState.pushConstantRanges.emplace_back(stageFlags, offset, size);
    executableState.pushConstantSize = std::max(executableState.pushConstantSize, offset + size);
}

/*******************************************************************************
 * Finalization
 *******************************************************************************/

Workload WorkloadBuilder::finish() {
    // Single-use guard
    if (workloadState_ == nullptr) {
        throw std::logic_error("WorkloadBuilder::finish() can only be called once");
    }

    // Public resource index cache
    workloadState_->publicResourceIndices.clear();
    workloadState_->publicResourceIndices.reserve(workloadState_->resources.size());
    for (uint32_t resourceIndex = 0; resourceIndex < workloadState_->resources.size(); ++resourceIndex) {
        if (isPublicRole(workloadState_->resources[resourceIndex].role)) {
            workloadState_->publicResourceIndices.push_back(resourceIndex);
        }
    }

    // Placeholder module index cache
    workloadState_->placeholderModuleIndices.clear();
    workloadState_->placeholderModuleIndices.reserve(workloadState_->modules.size());
    for (uint32_t moduleIndex = 0; moduleIndex < workloadState_->modules.size(); ++moduleIndex) {
        if (workloadState_->modules[moduleIndex].codeKind == ModuleCodeKind::Missing) {
            workloadState_->placeholderModuleIndices.push_back(moduleIndex);
        }
    }

    // Workload creation
    return Workload(std::move(workloadState_));
}

/*******************************************************************************
 * Shared helpers
 *******************************************************************************/

Resource::Role WorkloadBuilder::publicRoleForAccess(ResourceAccess access) {
    switch (access) {
    case ResourceAccess::Read:
        return Resource::Role::Input;
    case ResourceAccess::Write:
    case ResourceAccess::ReadWrite:
        return Resource::Role::Output;
    }
    throw std::runtime_error("Unsupported workload resource access");
}

} // namespace mlworkloadlib::detail
