/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mlworkloadlib {

/*******************************************************************************
 * Library feature queries
 *******************************************************************************/

enum class Feature {
    GlslModules,
    HlslModules,
};

// Return whether an optional library feature is available.
bool supports(Feature feature);

/*******************************************************************************
 * Common workload metadata
 *******************************************************************************/

enum class ExecutableKind {
    Graph,
    Compute,
};

enum class ResourceKind {
    Unknown,
    Tensor,
    StorageBuffer,
    Image,
};

enum class ResourceAccess {
    Read,
    Write,
    ReadWrite,
};

enum class ModuleCodeKind {
    Missing,
    Spirv,
    Glsl,
    Hlsl,
};

/*******************************************************************************
 * Workload construction inputs
 *******************************************************************************/

struct ModuleImplementation {
    ModuleCodeKind codeKind = ModuleCodeKind::Missing;
    std::vector<uint32_t> spirv;
    std::string source;
    std::string buildOptions;
    std::vector<std::filesystem::path> includeDirs;
};

struct DispatchShape {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

struct SpecializationInfo {
    std::vector<vk::SpecializationMapEntry> mapEntries;
    std::vector<uint8_t> data;

    bool empty() const noexcept { return mapEntries.empty(); }
};

struct TensorRequirements {
    vk::TensorUsageFlagsARM usage;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
};

struct BufferRequirements {
    vk::BufferUsageFlags usage;
    vk::DeviceSize byteSize = 0;
};

struct SamplerRequirements {
    vk::Filter magFilter = vk::Filter::eNearest;
    vk::Filter minFilter = vk::Filter::eNearest;
    vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eNearest;
    vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eClampToEdge;
    vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eClampToEdge;
    vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eClampToEdge;
};

struct ImageRequirements {
    vk::ImageUsageFlags usage;
    vk::Extent3D extent;
    vk::ImageLayout requiredLayout{};
    vk::ImageSubresourceRange range;
    std::optional<SamplerRequirements> runtimeSampler;
};

struct ResourceRequirements {
    ResourceKind kind = ResourceKind::Unknown;
    vk::DescriptorType descriptorType = {};
    vk::Format format = vk::Format::eUndefined;
    vk::DeviceSize elementCount = 0;
    bool requiresBoundMemoryInfo = false;
    TensorRequirements tensor;
    BufferRequirements buffer;
    ImageRequirements image;
};

/*******************************************************************************
 * Standalone compute workload descriptions
 *******************************************************************************/

struct ComputeShaderResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    ResourceAccess access = ResourceAccess::ReadWrite;
    ResourceRequirements resource;
};

struct ComputeShaderDescription {
    ModuleImplementation module;
    std::string entryPoint = "main";
    DispatchShape dispatch;
    uint32_t pushConstantSize = 0;
    SpecializationInfo specializationInfo;
    bool implicitBarrier = true;
    std::vector<ComputeShaderResource> resources;
};

/*******************************************************************************
 * Standalone data graph workload descriptions
 *******************************************************************************/

struct DataGraphResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    ResourceAccess access = ResourceAccess::ReadWrite;
    ResourceRequirements resource;
};

struct DataGraphConstant {
    struct Sparsity {
        int64_t dimension = 0;
    };

    std::string name;
    ResourceRequirements resource;

    // Borrowed payload. Caller keeps it valid for the Workload lifetime.
    const void *data = nullptr;
    std::uint64_t size = 0;

    std::optional<Sparsity> sparse2To4 = std::nullopt;
};

struct DataGraphPipelineMetadata {
    std::string identifier;
    vk::PipelineCreateFlags2 flags;
    SpecializationInfo specializationInfo;
};

struct DataGraphDescription {
    ModuleImplementation module;
    std::string entryPoint = "main";
    std::vector<DataGraphResource> resources;
    std::vector<DataGraphConstant> constants;
    DataGraphPipelineMetadata pipeline;
    bool implicitBarrier = true;
};

} // namespace mlworkloadlib
