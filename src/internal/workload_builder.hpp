/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "workload_impl.hpp"

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlworkloadlib::detail {

/*******************************************************************************
 * Module metadata
 *******************************************************************************/

Module moduleFromImplementation(ModuleImplementation implementation, std::string name, std::string entryPoint);

/*******************************************************************************
 * Workload builder
 *******************************************************************************/

class WorkloadBuilder {
  public:
    /***************************************************************************
     * Lifetime
     **************************************************************************/

    WorkloadBuilder();
    explicit WorkloadBuilder(std::unique_ptr<MemoryMap> mappedFile);

    /***************************************************************************
     * Resources
     **************************************************************************/

    void reserveResources(std::size_t count);
    uint32_t addResource(std::string name, const ResourceRequirements &requirements, Resource::Role role,
                         std::optional<uint32_t> aliasGroupId = std::nullopt);
    uint32_t addTensorResource(std::string name, Resource::Role role, std::optional<vk::DescriptorType> descriptorType,
                               vk::Format format, std::vector<int64_t> shape, std::vector<int64_t> stride,
                               vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId,
                               bool requiresBoundMemoryInfo, vk::TensorUsageFlagsARM usage);
    uint32_t addBufferResource(std::string name, Resource::Role role, vk::DescriptorType descriptorType,
                               vk::Format format, std::vector<int64_t> shape, std::vector<int64_t> stride,
                               vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId,
                               bool requiresBoundMemoryInfo, vk::DeviceSize byteSize, vk::BufferUsageFlags usage);
    uint32_t addImageResource(std::string name, Resource::Role role, vk::DescriptorType descriptorType,
                              vk::Format format, std::vector<int64_t> shape, std::vector<int64_t> stride,
                              vk::DeviceSize elementCount, std::optional<uint32_t> aliasGroupId,
                              bool requiresBoundMemoryInfo, vk::Extent3D extent, vk::ImageUsageFlags usage,
                              vk::ImageLayout layout, vk::ImageSubresourceRange subresourceRange,
                              std::optional<Resource::ImageMetadata::SamplerConfig> samplerConfig);
    Resource &resource(uint32_t resourceIndex);
    const Resource &resource(uint32_t resourceIndex) const;
    void setPublicResourceName(uint32_t resourceIndex, std::string name);

    /***************************************************************************
     * Modules
     **************************************************************************/

    void reserveModules(std::size_t count);
    uint32_t addModule(std::string name, std::string entryPoint, ModuleCodeKind codeKind, std::vector<uint32_t> code,
                       std::string source = {}, std::string buildOptions = {},
                       std::vector<std::filesystem::path> includeDirs = {});
    uint32_t addModule(ModuleImplementation implementation, std::string name, std::string entryPoint);

    /***************************************************************************
     * Constants
     **************************************************************************/

    void reserveConstants(std::size_t count);
    uint32_t addConstantResource(std::string name, const ResourceRequirements &requirements,
                                 std::optional<uint32_t> aliasGroupId = std::nullopt);
    uint32_t addConstant(uint32_t resourceIndex, const void *data, std::uint64_t size, int64_t sparsityDimension);

    /***************************************************************************
     * Executables
     **************************************************************************/

    void reserveExecutables(std::size_t count);
    uint32_t addExecutable(std::string name, ExecutableKind type, uint32_t moduleIndex);
    void addDescriptorBinding(uint32_t executableIndex, uint32_t resourceIndex, uint32_t set, uint32_t binding,
                              ResourceAccess access);
    void addPushConstantRange(uint32_t executableIndex, vk::ShaderStageFlags stageFlags, uint32_t offset,
                              uint32_t size);
    Executable &executable(uint32_t executableIndex);
    const Executable &executable(uint32_t executableIndex) const;

    void setDispatchShape(uint32_t executableIndex, DispatchShape dispatch);
    void setDispatchShape(uint32_t executableIndex, std::array<uint32_t, 3> dispatch);
    void setImplicitBarrier(uint32_t executableIndex, bool implicitBarrier);
    void setSpecializationInfo(uint32_t executableIndex, SpecializationInfo specializationInfo);

    /***************************************************************************
     * Finalization
     **************************************************************************/

    // Finalizes and transfers ownership of the accumulated workload state.
    // The builder must not be used again after finish().
    Workload finish();

    /***************************************************************************
     * Shared helpers
     **************************************************************************/

    static Resource::Role publicRoleForAccess(ResourceAccess access);

  private:
    /***************************************************************************
     * Append helpers
     **************************************************************************/

    uint32_t appendResource(Resource workloadResource);
    uint32_t appendModule(Module workloadModule);
    uint32_t appendConstant(Constant workloadConstant);

    /***************************************************************************
     * Stored state
     **************************************************************************/

    std::unique_ptr<Workload::Impl> workloadState_;
};
} // namespace mlworkloadlib::detail
