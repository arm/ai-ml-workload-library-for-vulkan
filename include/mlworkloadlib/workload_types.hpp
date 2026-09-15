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

namespace mlsdk::workloadlib {

/*******************************************************************************
 * Library feature queries
 *******************************************************************************/

/** @brief Optional capabilities that may be compiled into the library. */
enum class Feature {
    GlslModules,
    HlslModules,
};

/** @brief Returns whether an optional library capability is available. */
bool supports(Feature feature);

/*******************************************************************************
 * Common workload metadata
 *******************************************************************************/

/** @brief Kind of executable represented by workload metadata. */
enum class ExecutableKind {
    Graph,
    Compute,
};

/** @brief Public workload resource category. */
enum class ResourceKind {
    Unknown,
    Tensor,
    StorageBuffer,
    Image,
};

/** @brief Access performed by an executable on a resource. */
enum class ResourceAccess {
    Read,
    Write,
    ReadWrite,
};

/** @brief Representation used for executable module code. */
enum class ModuleCodeKind {
    Missing,
    Spirv,
    Glsl,
    Hlsl,
};

/*******************************************************************************
 * Workload construction inputs
 *******************************************************************************/

/**
 * @brief Module code and compilation options used to construct or complete a workload.
 *
 * Exactly one of @ref spirv or @ref source is consumed according to
 * @ref codeKind. GLSL and HLSL require the corresponding Feature to be
 * available when the Session is configured.
 */
struct ModuleImplementation {
    ModuleCodeKind codeKind = ModuleCodeKind::Missing;
    std::vector<uint32_t> spirv;
    std::string source;
    std::string buildOptions;
    std::vector<std::filesystem::path> includeDirs;
};

/** @brief Compute dispatch dimensions in workgroups. */
struct DispatchShape {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

/**
 * @brief Vulkan specialization-constant map entries and their packed data.
 *
 * Each map entry's offset and size identify bytes in @ref data. Constant IDs
 * must be unique and every entry must fit within the data buffer.
 */
struct SpecializationInfo {
    std::vector<vk::SpecializationMapEntry> mapEntries;
    std::vector<uint8_t> data;

    bool empty() const noexcept { return mapEntries.empty(); }
};

/** @brief Tensor-specific resource creation requirements. */
struct TensorRequirements {
    vk::TensorUsageFlagsARM usage;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
};

/** @brief Storage-buffer-specific resource creation requirements. */
struct BufferRequirements {
    vk::BufferUsageFlags usage;
    vk::DeviceSize byteSize = 0;
};

/** @brief Parameters used when the runtime creates a sampler for an image. */
struct SamplerRequirements {
    vk::Filter magFilter = vk::Filter::eNearest;
    vk::Filter minFilter = vk::Filter::eNearest;
    vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eNearest;
    vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eClampToEdge;
    vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eClampToEdge;
    vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eClampToEdge;
};

/** @brief Image-specific descriptor and resource creation requirements. */
struct ImageRequirements {
    vk::ImageUsageFlags usage;
    vk::Extent3D extent;
    vk::ImageLayout requiredLayout{};
    vk::ImageSubresourceRange range;
    std::optional<SamplerRequirements> runtimeSampler;
};

/**
 * @brief Common and kind-specific requirements for a workload resource.
 *
 * @ref kind may be inferred from a supported @ref descriptorType. Conversely,
 * tensor and storage-buffer descriptor types may be inferred from @ref kind.
 * Image resources must explicitly select a sampled- or storage-image descriptor
 * type. Only the kind-specific member matching @ref kind is consumed.
 */
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

/** @brief Public storage resource and descriptor binding for a compute workload. */
struct ComputeShaderResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    ResourceAccess access = ResourceAccess::ReadWrite;
    ResourceRequirements resource;
};

/** @brief Description used to build a standalone compute workload. */
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

/** @brief Public storage resource and descriptor binding for a data graph workload. */
struct DataGraphResource {
    std::string name;
    uint32_t set = 0;
    uint32_t binding = 0;
    ResourceAccess access = ResourceAccess::ReadWrite;
    ResourceRequirements resource;
};

/**
 * @brief Constant tensor supplied to a standalone data graph workload.
 *
 * Constant payloads are borrowed rather than copied. The payload must remain
 * valid for the lifetime of the Workload created from the description.
 */
struct DataGraphConstant {
    struct Sparsity {
        int64_t dimension = 0;
    };

    std::string name;
    ResourceRequirements resource;

    const void *data = nullptr;
    std::uint64_t size = 0;

    std::optional<Sparsity> sparse2To4 = std::nullopt;
};

/** @brief Vulkan data graph pipeline metadata. */
struct DataGraphPipelineMetadata {
    std::string identifier;
    vk::PipelineCreateFlags2 flags;
    SpecializationInfo specializationInfo;
};

/** @brief Description used to build a standalone data graph workload. */
struct DataGraphDescription {
    ModuleImplementation module;
    std::string entryPoint = "main";
    std::vector<DataGraphResource> resources;
    std::vector<DataGraphConstant> constants;
    DataGraphPipelineMetadata pipeline;
    bool implicitBarrier = true;
};

} // namespace mlsdk::workloadlib
