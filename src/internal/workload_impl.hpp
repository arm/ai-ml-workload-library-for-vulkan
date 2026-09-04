/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/workload.hpp"

#include "vgf-utils/memory_map.hpp"

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlworkloadlib::detail {

/*******************************************************************************
 * Workload metadata model
 *******************************************************************************/

struct DescriptorBinding {
    uint32_t resourceIndex = 0;
    uint32_t set = 0;
    uint32_t binding = 0;
    ResourceAccess access = ResourceAccess::ReadWrite;
};

struct Resource {
    enum class Role {
        Input,
        Output,
        Intermediate,
        Constant,
    };

    struct TensorMetadata {
        vk::TensorUsageFlagsARM usage;
    };

    struct BufferMetadata {
        vk::DeviceSize byteSize = 0;
        vk::BufferUsageFlags usage;
    };

    struct ImageMetadata {
        struct SamplerConfig {
            vk::Filter minFilter = vk::Filter::eNearest;
            vk::Filter magFilter = vk::Filter::eNearest;
            vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eNearest;
            vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eClampToEdge;
            vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eClampToEdge;
            vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eClampToEdge;
            vk::BorderColor borderColor = vk::BorderColor::eFloatTransparentBlack;
        };

        vk::Extent3D extent;
        vk::ImageUsageFlags usage;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        vk::ImageSubresourceRange subresourceRange;
        std::optional<SamplerConfig> samplerConfig;
    };

    std::string name;
    Role role = Role::Input;
    std::optional<vk::DescriptorType> descriptorType;
    vk::Format format = vk::Format::eUndefined;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    vk::DeviceSize elementCount = 0;
    std::optional<uint32_t> aliasGroupId;
    std::variant<std::monostate, TensorMetadata, BufferMetadata, ImageMetadata> metadata;
    bool requiresBoundMemoryInfo = false;
};

inline const Resource::TensorMetadata &tensorMetadata(const Resource &resource) {
    return std::get<Resource::TensorMetadata>(resource.metadata);
}

inline const Resource::BufferMetadata &bufferMetadata(const Resource &resource) {
    return std::get<Resource::BufferMetadata>(resource.metadata);
}

inline const Resource::ImageMetadata &imageMetadata(const Resource &resource) {
    return std::get<Resource::ImageMetadata>(resource.metadata);
}

struct Module {
    std::string name;
    std::string entryPoint;
    ModuleCodeKind codeKind = ModuleCodeKind::Spirv;
    std::vector<uint32_t> code;
    std::string source;
    std::string buildOptions;
    std::vector<std::filesystem::path> includeDirs;
};

struct Constant {
    uint32_t resourceIndex = 0;
    ArrayView<const uint8_t> payloadView;
    int64_t sparsityDimension = -1;
};

struct Executable {
    uint32_t executableIndex = 0;
    std::string name;
    ExecutableKind type = ExecutableKind::Graph;
    uint32_t moduleIndex = 0;
    std::vector<DescriptorBinding> bindings;
    std::array<uint32_t, 3> dispatchShape = {};
    std::vector<uint32_t> constantIndexes;
    std::string dataGraphPipelineIdentifier;
    vk::PipelineCreateFlags2 dataGraphPipelineFlags;
    SpecializationInfo specializationInfo;
    std::vector<vk::PushConstantRange> pushConstantRanges;
    uint32_t pushConstantSize = 0;
    bool implicitBarrier = true;
};

} // namespace mlworkloadlib::detail

namespace mlworkloadlib {

/*******************************************************************************
 * Implementation state
 *******************************************************************************/

struct Workload::Impl {
    using Constant = detail::Constant;
    using DescriptorBinding = detail::DescriptorBinding;
    using Executable = detail::Executable;
    using Module = detail::Module;
    using Resource = detail::Resource;

    /***************************************************************************
     * Stored metadata
     **************************************************************************/

    std::unique_ptr<MemoryMap> mappedFile;

    std::vector<Resource> resources;
    std::vector<uint32_t> publicResourceIndices;

    std::vector<Module> modules;

    std::vector<uint32_t> placeholderModuleIndices;

    std::vector<Constant> constants;
    std::vector<Executable> executables;
};

} // namespace mlworkloadlib
