/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib/workload.hpp"

#include "internal/utils.hpp"
#include "internal/workload_builder.hpp"

#include "vgf/decoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlworkloadlib {
namespace vgflib = mlsdk::vgflib;
namespace utils = detail::utils;
namespace vulkan_helpers = detail::vulkan_helpers;

using Resource = detail::Resource;
using WorkloadBuilder = detail::WorkloadBuilder;

namespace {

/*******************************************************************************
 * VGF decoding helpers
 *******************************************************************************/

template <typename Size, typename T> Size checkedDataViewByteSize(vgflib::DataView<T> view) {
    const auto elementCount = view.size();
    if (elementCount == 0) {
        return 0;
    }
    if (view.data() == nullptr) {
        throw std::runtime_error("VGF data view has null data");
    }
    if (elementCount > std::numeric_limits<Size>::max() / sizeof(T)) {
        throw std::runtime_error("VGF data view size is too large");
    }
    return static_cast<Size>(elementCount) * static_cast<Size>(sizeof(T));
}

template <typename T> std::vector<T> toVector(vgflib::DataView<T> view) {
    static_assert(std::is_trivially_copyable_v<T>, "VGF DataView copy requires trivially copyable elements");

    if (checkedDataViewByteSize<std::size_t>(view) == 0) {
        return {};
    }

    if (reinterpret_cast<std::uintptr_t>(view.data()) % alignof(T) != 0) {
        throw std::runtime_error("VGF data view is misaligned");
    }

    return {view.begin(), view.end()};
}

template <typename T> std::uint64_t byteSize(vgflib::DataView<T> view) {
    return checkedDataViewByteSize<std::uint64_t>(view);
}

ExecutableKind executableKind(vgflib::ModuleType type) {
    switch (type) {
    case vgflib::ModuleType::GRAPH:
        return ExecutableKind::Graph;
    case vgflib::ModuleType::COMPUTE:
        return ExecutableKind::Compute;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

Resource::Role roleForResource(vgflib::ResourceCategory category) {
    switch (category) {
    case vgflib::ResourceCategory::INPUT:
        return Resource::Role::Input;
    case vgflib::ResourceCategory::OUTPUT:
        return Resource::Role::Output;
    case vgflib::ResourceCategory::INTERMEDIATE:
        return Resource::Role::Intermediate;
    case vgflib::ResourceCategory::CONSTANT:
        return Resource::Role::Constant;
    default:
        throw std::runtime_error("Unsupported VGF resource category");
    }
}

std::optional<vk::DescriptorType> descriptorTypeForResource(const vgflib::ModelResourceTableDecoder &decoder,
                                                            uint32_t resourceIndex) {
    if (const auto descriptorType = decoder.getDescriptorType(resourceIndex)) {
        return vk::DescriptorType(*descriptorType);
    }
    return std::nullopt;
}

ResourceKind kindForResource(std::optional<vk::DescriptorType> descriptorType, Resource::Role role) {
    if (!descriptorType.has_value()) {
        if (role == Resource::Role::Constant) {
            return ResourceKind::Tensor;
        }
        throw std::runtime_error("VGF non-constant resource is missing descriptor type");
    }

    const auto kind = vulkan_helpers::resourceKind(*descriptorType);
    if (kind == ResourceKind::Unknown) {
        throw std::runtime_error("Unsupported VGF resource descriptor type");
    }
    return kind;
}

std::optional<Resource::ImageMetadata::SamplerConfig>
samplerConfigFromVgfResource(const vgflib::ModelResourceTableDecoder &decoder, uint32_t resourceIndex) {
    const auto *const samplerConfigHandle = decoder.getSamplerConfigHandle(resourceIndex);
    if (samplerConfigHandle == nullptr) {
        return std::nullopt;
    }

    return Resource::ImageMetadata::SamplerConfig{
        vk::Filter(decoder.getSamplerConfigMinFilter(samplerConfigHandle)),
        vk::Filter(decoder.getSamplerConfigMagFilter(samplerConfigHandle)),
        vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode(decoder.getSamplerConfigAddressModeU(samplerConfigHandle)),
        vk::SamplerAddressMode(decoder.getSamplerConfigAddressModeV(samplerConfigHandle)),
        vk::SamplerAddressMode::eClampToEdge,
        vk::BorderColor(decoder.getSamplerConfigBorderColor(samplerConfigHandle))};
}

/*******************************************************************************
 * VGF workload metadata
 *******************************************************************************/

uint32_t addDecodedConstantResource(WorkloadBuilder &builder, vk::Format format, std::vector<int64_t> shape,
                                    std::vector<int64_t> stride, vk::DeviceSize elementCount,
                                    std::optional<uint32_t> aliasGroupId, bool requiresBoundMemoryInfo) {
    // VGF constants are graph pipeline constants, not descriptor-bound
    // resources. They still need tensor metadata for
    // vk::DataGraphPipelineConstantARM creation.
    return builder.addTensorResource({}, Resource::Role::Constant, std::nullopt, format, std::move(shape),
                                     std::move(stride), elementCount, aliasGroupId, requiresBoundMemoryInfo,
                                     vk::TensorUsageFlagBitsARM::eDataGraph);
}

uint32_t addDecodedResource(WorkloadBuilder &builder, const vgflib::ModelResourceTableDecoder &decoder,
                            uint32_t resourceIndex) {
    const auto role = roleForResource(decoder.getCategory(resourceIndex));
    const auto descriptorType = descriptorTypeForResource(decoder, resourceIndex);
    const auto kind = kindForResource(descriptorType, role);
    const auto format = vk::Format(decoder.getVkFormat(resourceIndex));
    auto shape = toVector(decoder.getTensorShape(resourceIndex));
    auto stride = toVector(decoder.getTensorStride(resourceIndex));
    const auto elementCount = utils::elementCount(shape);
    const auto aliasGroupId = decoder.getAliasGroupId(resourceIndex);
    const bool requiresBoundMemoryInfo = aliasGroupId.has_value();

    if (!descriptorType.has_value()) {
        return addDecodedConstantResource(builder, format, std::move(shape), std::move(stride), elementCount,
                                          aliasGroupId, requiresBoundMemoryInfo);
    }

    switch (kind) {
    case ResourceKind::Tensor:
        return builder.addTensorResource({}, role, descriptorType, format, std::move(shape), std::move(stride),
                                         elementCount, aliasGroupId, requiresBoundMemoryInfo,
                                         vk::TensorUsageFlagBitsARM::eDataGraph);
    case ResourceKind::StorageBuffer:
        return builder.addBufferResource({}, role, *descriptorType, format, std::move(shape), std::move(stride),
                                         elementCount, aliasGroupId, requiresBoundMemoryInfo, 0,
                                         vk::BufferUsageFlagBits::eStorageBuffer);
    case ResourceKind::Image:
        return builder.addImageResource(
            {}, role, *descriptorType, format, std::move(shape), std::move(stride), elementCount, aliasGroupId,
            requiresBoundMemoryInfo, {}, vulkan_helpers::imageUsage(*descriptorType, aliasGroupId.has_value()),
            vulkan_helpers::imageLayout(*descriptorType), {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            samplerConfigFromVgfResource(decoder, resourceIndex));
    case ResourceKind::Unknown:
        throw std::runtime_error("Unsupported VGF resource descriptor type");
    }
    throw std::runtime_error("Unsupported VGF resource kind " + std::string(utils::resourceKindName(kind)));
}

uint32_t addDecodedModule(WorkloadBuilder &builder, const vgflib::ModuleTableDecoder &decoder, uint32_t moduleIndex) {
    auto name = std::string(decoder.getModuleName(moduleIndex));
    auto entryPoint = std::string(decoder.getModuleEntryPoint(moduleIndex));
    if (decoder.hasSPIRVCode(moduleIndex)) {
        return builder.addModule(std::move(name), std::move(entryPoint), ModuleCodeKind::Spirv,
                                 toVector(decoder.getSPIRVModuleCode(moduleIndex)));
    }
    if (decoder.hasGLSLCode(moduleIndex)) {
        return builder.addModule(std::move(name), std::move(entryPoint), ModuleCodeKind::Glsl, {},
                                 std::string(decoder.getGLSLModuleCode(moduleIndex)));
    }
    if (decoder.hasHLSLCode(moduleIndex)) {
        return builder.addModule(std::move(name), std::move(entryPoint), ModuleCodeKind::Hlsl, {},
                                 std::string(decoder.getHLSLModuleCode(moduleIndex)));
    }
    return builder.addModule(std::move(name), std::move(entryPoint), ModuleCodeKind::Missing, {});
}

uint32_t addDecodedConstant(WorkloadBuilder &builder, const vgflib::ConstantDecoder &decoder, uint32_t constantIndex) {
    const auto sparsityDimension = decoder.getConstantSparsityDimension(constantIndex);
    if (!utils::isSparsityDimensionValid(sparsityDimension)) {
        throw std::runtime_error("VGF constant has invalid sparsity dimension");
    }
    const auto constantData = decoder.getConstant(constantIndex);
    return builder.addConstant(decoder.getConstantMrtIndex(constantIndex), constantData.data(), byteSize(constantData),
                               sparsityDimension);
}

ResourceAccess accessForRole(Resource::Role role) {
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

bool bindingSlotMatches(const vgflib::ModelSequenceTableDecoder &decoder, uint32_t executableIndex,
                        uint32_t resourceIndex, uint32_t binding, bool input) {
    const auto *handle = input ? decoder.getSegmentInputBindingSlotsHandle(executableIndex)
                               : decoder.getSegmentOutputBindingSlotsHandle(executableIndex);
    if (handle == nullptr) {
        return false;
    }
    for (uint32_t slot = 0; slot < decoder.getBindingsSize(handle); ++slot) {
        if (decoder.getBindingSlotMrtIndex(handle, slot) == resourceIndex &&
            decoder.getBindingSlotBinding(handle, slot) == binding) {
            return true;
        }
    }
    return false;
}

ResourceAccess bindingAccess(const vgflib::ModelSequenceTableDecoder &decoder, uint32_t executableIndex,
                             uint32_t resourceIndex, uint32_t binding, Resource::Role role) {
    const bool input = bindingSlotMatches(decoder, executableIndex, resourceIndex, binding, true);
    const bool output = bindingSlotMatches(decoder, executableIndex, resourceIndex, binding, false);
    if (input && output) {
        return ResourceAccess::ReadWrite;
    }
    if (input) {
        return ResourceAccess::Read;
    }
    if (output) {
        return ResourceAccess::Write;
    }
    return accessForRole(role);
}

void addDecodedPushConstantRanges(const vgflib::ModelSequenceTableDecoder &decoder, WorkloadBuilder &builder,
                                  uint32_t executableIndex) {
    const auto *handle = decoder.getSegmentPushConstRange(executableIndex);
    if (handle == nullptr) {
        throw std::runtime_error("VGF segment push constant ranges are missing");
    }

    for (uint32_t rangeIndex = 0; rangeIndex < decoder.getPushConstRangesSize(handle); ++rangeIndex) {
        const auto stageFlags = vk::ShaderStageFlags(decoder.getPushConstRangeStageFlags(handle, rangeIndex));
        const auto offset = decoder.getPushConstRangeOffset(handle, rangeIndex);
        const auto size = decoder.getPushConstRangeSize(handle, rangeIndex);
        builder.addPushConstantRange(executableIndex, stageFlags, offset, size);
    }
}

void addDecodedDescriptorBinding(const vgflib::ModelSequenceTableDecoder &sequenceDecoder, WorkloadBuilder &builder,
                                 uint32_t executableIndex, uint32_t set, uint32_t resourceIndex, uint32_t binding) {
    const auto &resource = builder.resource(resourceIndex);
    if (!resource.descriptorType.has_value()) {
        throw std::runtime_error("Descriptor binding references a VGF resource without descriptor type");
    }
    builder.addDescriptorBinding(
        executableIndex, resourceIndex, set, binding,
        bindingAccess(sequenceDecoder, executableIndex, resourceIndex, binding, resource.role));
}

void populateVgfWorkload(WorkloadBuilder &builder, const void *data, std::size_t size) {
    // Header and tables
    auto header = vgflib::CreateHeaderDecoder(data, vgflib::HeaderSize(), size);
    if (!header) {
        throw std::runtime_error("Failed to decode VGF header");
    }

    const auto *bytes = static_cast<const std::byte *>(data);
    auto moduleTable =
        vgflib::CreateModuleTableDecoder(bytes + header->GetModuleTableOffset(), header->GetModuleTableSize());
    auto modelSequenceTable = vgflib::CreateModelSequenceTableDecoder(bytes + header->GetModelSequenceTableOffset(),
                                                                      header->GetModelSequenceTableSize());
    auto modelResourceTable = vgflib::CreateModelResourceTableDecoder(bytes + header->GetModelResourceTableOffset(),
                                                                      header->GetModelResourceTableSize());
    auto constantDecoder =
        vgflib::CreateConstantDecoder(bytes + header->GetConstantsOffset(), header->GetConstantsSize());
    if (!moduleTable || !modelSequenceTable || !modelResourceTable || !constantDecoder) {
        throw std::runtime_error("Failed to decode VGF tables");
    }

    // Resources
    builder.reserveResources(modelResourceTable->size());
    for (uint32_t resourceIndex = 0; resourceIndex < modelResourceTable->size(); ++resourceIndex) {
        if (addDecodedResource(builder, *modelResourceTable, resourceIndex) != resourceIndex) {
            throw std::logic_error("VGF resource index does not match builder insertion order");
        }
    }

    // Modules
    builder.reserveModules(moduleTable->size());
    for (uint32_t moduleIndex = 0; moduleIndex < moduleTable->size(); ++moduleIndex) {
        if (addDecodedModule(builder, *moduleTable, moduleIndex) != moduleIndex) {
            throw std::logic_error("VGF module index does not match builder insertion order");
        }
    }

    // Constants
    builder.reserveConstants(constantDecoder->size());
    for (uint32_t constantIndex = 0; constantIndex < constantDecoder->size(); ++constantIndex) {
        const uint32_t resourceIndex = constantDecoder->getConstantMrtIndex(constantIndex);
        if (builder.resource(resourceIndex).role != Resource::Role::Constant) {
            throw std::runtime_error("VGF constant references a non-constant resource");
        }
        if (addDecodedConstant(builder, *constantDecoder, constantIndex) != constantIndex) {
            throw std::logic_error("VGF constant index does not match builder insertion order");
        }
    }

    // Executables and descriptor bindings
    // Model-sequence entries (segments) become workload executables in sequence order.
    builder.reserveExecutables(modelSequenceTable->modelSequenceTableSize());
    for (uint32_t executableIndex = 0; executableIndex < modelSequenceTable->modelSequenceTableSize();
         ++executableIndex) {
        const uint32_t moduleIndex = modelSequenceTable->getSegmentModuleIndex(executableIndex);
        const auto addedExecutableIndex =
            builder.addExecutable(std::string(modelSequenceTable->getSegmentName(executableIndex)),
                                  executableKind(modelSequenceTable->getSegmentType(executableIndex)), moduleIndex);
        if (addedExecutableIndex != executableIndex) {
            throw std::logic_error("VGF executable index does not match builder insertion order");
        }
        auto &executable = builder.executable(executableIndex);

        const auto dispatchShape = modelSequenceTable->getSegmentDispatchShape(executableIndex);
        if (dispatchShape.size() > executable.dispatchShape.size()) {
            throw std::runtime_error("VGF executable dispatch shape has more than three dimensions");
        }
        std::copy(dispatchShape.begin(), dispatchShape.end(), executable.dispatchShape.begin());
        addDecodedPushConstantRanges(*modelSequenceTable, builder, executableIndex);

        for (const auto constantIndex : modelSequenceTable->getSegmentConstantIndexes(executableIndex)) {
            executable.constantIndexes.push_back(constantIndex);
        }

        for (uint32_t descIdx = 0; descIdx < modelSequenceTable->getSegmentDescriptorSetInfosSize(executableIndex);
             ++descIdx) {
            const uint32_t set = modelSequenceTable->getSegmentDescriptorSetIndex(executableIndex, descIdx);
            const auto *handle = modelSequenceTable->getDescriptorBindingSlotsHandle(executableIndex, descIdx);
            if (handle == nullptr) {
                continue;
            }
            for (uint32_t slot = 0; slot < modelSequenceTable->getBindingsSize(handle); ++slot) {
                const uint32_t resourceIndex = modelSequenceTable->getBindingSlotMrtIndex(handle, slot);
                const uint32_t binding = modelSequenceTable->getBindingSlotBinding(handle, slot);
                addDecodedDescriptorBinding(*modelSequenceTable, builder, executableIndex, set, resourceIndex, binding);
            }
        }
    }

    // Public resource names
    auto addNames = [&](vgflib::BindingSlotArrayHandle bindings, vgflib::NameArrayHandle names) {
        if (bindings == nullptr || names == nullptr) {
            return;
        }
        const auto count =
            std::min(modelSequenceTable->getBindingsSize(bindings), modelSequenceTable->getNamesSize(names));
        for (uint32_t slot = 0; slot < count; ++slot) {
            builder.setPublicResourceName(modelSequenceTable->getBindingSlotMrtIndex(bindings, slot),
                                          std::string(modelSequenceTable->getName(names, slot)));
        }
    };
    addNames(modelSequenceTable->getModelSequenceInputBindingSlotsHandle(),
             modelSequenceTable->getModelSequenceInputNamesHandle());
    addNames(modelSequenceTable->getModelSequenceOutputBindingSlotsHandle(),
             modelSequenceTable->getModelSequenceOutputNamesHandle());
}

Workload decodeVgfMemory(std::unique_ptr<MemoryMap> mappedFile) {
    if (mappedFile == nullptr) {
        throw std::logic_error("File-backed VGF decode requires a memory map");
    }

    const auto *data = mappedFile->ptr();
    const auto size = mappedFile->size();

    WorkloadBuilder builder(std::move(mappedFile));
    populateVgfWorkload(builder, data, size);
    return builder.finish();
}

Workload decodeVgfMemory(const void *data, std::size_t size) {
    WorkloadBuilder builder;
    populateVgfWorkload(builder, data, size);
    return builder.finish();
}

} // namespace

/*******************************************************************************
 * VGF workload
 *******************************************************************************/

Workload Workload::fromVGF(const std::filesystem::path &path) {
    return decodeVgfMemory(std::make_unique<MemoryMap>(path.string()));
}

Workload Workload::fromVGF(const void *data, std::size_t size) { return decodeVgfMemory(data, size); }

} // namespace mlworkloadlib
