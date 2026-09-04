/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/module_compiler.hpp"
#include "internal/utils.hpp"
#include "internal/workload_impl.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mlworkloadlib {

/*******************************************************************************
 * State access
 *******************************************************************************/

const Workload::Impl &Workload::workloadImpl() const noexcept { return *impl_; }

const Workload::Impl &workloadImpl(const Workload &workload) noexcept { return workload.workloadImpl(); }

namespace vulkan_helpers = detail::vulkan_helpers;
namespace utils = detail::utils;

using DescriptorBinding = detail::DescriptorBinding;
using Executable = detail::Executable;
using Module = detail::Module;
using Resource = detail::Resource;

namespace {

/*******************************************************************************
 * Internal helpers
 *******************************************************************************/

const std::vector<Resource> &workloadResources(const Workload &workload) noexcept {
    return workloadImpl(workload).resources;
}

const Resource &workloadResource(const Workload &workload, uint32_t resourceIndex) {
    return workloadResources(workload).at(resourceIndex);
}

const std::vector<Module> &workloadModules(const Workload &workload) noexcept { return workloadImpl(workload).modules; }

const Module &workloadModule(const Workload &workload, uint32_t moduleIndex) {
    return workloadModules(workload).at(moduleIndex);
}

const Executable &workloadExecutable(const Workload &workload, uint32_t executableIndex) {
    return workloadImpl(workload).executables.at(executableIndex);
}

uint32_t workloadPublicResourceCount(const Workload &workload) {
    return static_cast<uint32_t>(workloadImpl(workload).publicResourceIndices.size());
}

uint32_t internalResourceIndexForPublicResource(const Workload &workload, uint32_t publicResourceIndex) {
    return workloadImpl(workload).publicResourceIndices.at(publicResourceIndex);
}

uint32_t workloadExecutableCount(const Workload &workload) {
    return static_cast<uint32_t>(workloadImpl(workload).executables.size());
}

uint32_t workloadPlaceholderModuleCount(const Workload &workload) {
    return static_cast<uint32_t>(workloadImpl(workload).placeholderModuleIndices.size());
}

uint32_t moduleIndexForPlaceholder(const Workload &workload, uint32_t placeholderIndex) {
    return workloadImpl(workload).placeholderModuleIndices.at(placeholderIndex);
}

ResourceAccess workloadResourceAccess(const Workload &workload, uint32_t resourceIndex) {
    return utils::resourceAccess(workload, resourceIndex);
}

vk::DescriptorType workloadDescriptorType(const Workload &workload, const DescriptorBinding &descBinding) {
    return utils::descriptorType(workload, descBinding);
}

std::optional<uint32_t> publicResourceIndexForInternalResourceIndex(const Workload &workload,
                                                                    uint32_t internalResourceIndex) {
    const auto &publicResourceIndices = workloadImpl(workload).publicResourceIndices;
    const auto publicResource =
        std::find(publicResourceIndices.begin(), publicResourceIndices.end(), internalResourceIndex);
    if (publicResource == publicResourceIndices.end()) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(publicResource - publicResourceIndices.begin());
}

} // namespace

/*******************************************************************************
 * Feature support
 *******************************************************************************/

bool supports(Feature feature) {
    switch (feature) {
    case Feature::GlslModules:
        return detail::supportsGlslModules();
    case Feature::HlslModules:
        return detail::supportsHlslModules();
    }
    return false;
}

/*******************************************************************************
 * Lifetime
 *******************************************************************************/

Workload::Workload(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Workload::~Workload() = default;

/*******************************************************************************
 * Resource inspection
 *******************************************************************************/

uint32_t Workload::resourceCount() const { return workloadPublicResourceCount(*this); }

ResourceView Workload::resource(uint32_t publicResourceIndex) const {
    if (publicResourceIndex >= resourceCount()) {
        throw std::out_of_range("Workload resource index out of range");
    }
    return {this, publicResourceIndex, internalResourceIndexForPublicResource(*this, publicResourceIndex)};
}

ResourceView Workload::ResourceRange::Iterator::operator*() const { return workload()->resource(index()); }

Workload::ResourceRange::Iterator Workload::ResourceRange::begin() const { return {workload_, 0}; }

Workload::ResourceRange::Iterator Workload::ResourceRange::end() const {
    return {workload_, workload_->resourceCount()};
}

Workload::ResourceRange Workload::resources() const { return ResourceRange(this); }

/*******************************************************************************
 * Executable inspection
 *******************************************************************************/

uint32_t Workload::executableCount() const { return workloadExecutableCount(*this); }

ExecutableView Workload::executable(uint32_t executableIndex) const {
    if (executableIndex >= executableCount()) {
        throw std::out_of_range("Workload executable index out of range");
    }
    return {this, executableIndex};
}

ExecutableView Workload::ExecutableRange::Iterator::operator*() const { return {workload(), index()}; }

Workload::ExecutableRange::Iterator Workload::ExecutableRange::begin() const { return {workload_, 0}; }

Workload::ExecutableRange::Iterator Workload::ExecutableRange::end() const {
    return {workload_, workload_->executableCount()};
}

Workload::ExecutableRange Workload::executables() const { return ExecutableRange(this); }

/*******************************************************************************
 * Placeholder module inspection
 *******************************************************************************/

uint32_t Workload::placeholderModuleCount() const { return workloadPlaceholderModuleCount(*this); }

PlaceholderModuleView Workload::placeholderModule(uint32_t placeholderModuleIndex) const {
    return {this, placeholderModuleIndex, moduleIndexForPlaceholder(*this, placeholderModuleIndex)};
}

PlaceholderModuleView Workload::PlaceholderModuleRange::Iterator::operator*() const {
    return workload()->placeholderModule(index());
}

Workload::PlaceholderModuleRange::Iterator Workload::PlaceholderModuleRange::begin() const { return {workload_, 0}; }

Workload::PlaceholderModuleRange::Iterator Workload::PlaceholderModuleRange::end() const {
    return {workload_, workload_->placeholderModuleCount()};
}

Workload::PlaceholderModuleRange Workload::placeholderModules() const { return PlaceholderModuleRange(this); }

/*******************************************************************************
 * Resource requirement views
 *******************************************************************************/

ResourceKind ResourceRequirementsView::kind() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    return resource.descriptorType ? vulkan_helpers::resourceKind(*resource.descriptorType) : ResourceKind::Unknown;
}

vk::DescriptorType ResourceRequirementsView::descriptorType() const {
    return workloadResource(*workload_, resourceIndex_).descriptorType.value_or(vk::DescriptorType{});
}

vk::Format ResourceRequirementsView::format() const { return workloadResource(*workload_, resourceIndex_).format; }

bool ResourceRequirementsView::participatesInAliasing() const {
    return workloadResource(*workload_, resourceIndex_).aliasGroupId.has_value();
}

bool ResourceRequirementsView::requiresBoundMemoryInfo() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    return resource.requiresBoundMemoryInfo || resource.aliasGroupId.has_value();
}

vk::DeviceSize ResourceRequirementsView::elementCount() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    return resource.elementCount != 0 ? resource.elementCount : utils::elementCount(resource.shape);
}

vk::DeviceSize ResourceRequirementsView::byteSize() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    if (const auto *metadata = std::get_if<Resource::BufferMetadata>(&resource.metadata)) {
        if (metadata->byteSize != 0) {
            return metadata->byteSize;
        }
    }
    vk::DeviceSize elementSize = 0;
    switch (resource.format) {
    case vk::Format::eR8Sint:
        elementSize = 1;
        break;
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
    case vk::Format::eR8G8B8A8Snorm:
    case vk::Format::eR8G8B8A8Unorm:
        elementSize = 4;
        break;
    default:
        return 0;
    }

    if (!resource.stride.empty()) {
        vk::DeviceSize size = elementSize;
        for (uint32_t i = 0; i < resource.shape.size(); ++i) {
            if (resource.shape[i] <= 0) {
                return 0;
            }
            size +=
                static_cast<vk::DeviceSize>(resource.shape[i] - 1) * static_cast<vk::DeviceSize>(resource.stride[i]);
        }
        return size;
    }
    return utils::elementCount(resource.shape) * elementSize;
}

TensorRequirementsView ResourceRequirementsView::asTensor() const {
    if (kind() != ResourceKind::Tensor) {
        throw std::runtime_error("Workload resource is not a tensor");
    }
    return {workload_, resourceIndex_};
}

BufferRequirementsView ResourceRequirementsView::asBuffer() const {
    const auto resourceKind = kind();
    if (resourceKind != ResourceKind::StorageBuffer) {
        throw std::runtime_error("Workload resource is not a storage buffer; actual kind is " +
                                 std::string(utils::resourceKindName(resourceKind)));
    }
    return {workload_, resourceIndex_};
}

ImageRequirementsView ResourceRequirementsView::asImage() const {
    if (kind() != ResourceKind::Image) {
        throw std::runtime_error("Workload resource is not an image");
    }
    return {workload_, resourceIndex_};
}

/*******************************************************************************
 * Tensor requirement views
 *******************************************************************************/

vk::TensorUsageFlagsARM TensorRequirementsView::usage() const {
    return detail::tensorMetadata(workloadResource(*workload_, resourceIndex_)).usage;
}

ArrayView<const int64_t> TensorRequirementsView::shape() const {
    const auto &shape = workloadResource(*workload_, resourceIndex_).shape;
    return {shape.data(), shape.size()};
}

ArrayView<const int64_t> TensorRequirementsView::stride() const {
    const auto &stride = workloadResource(*workload_, resourceIndex_).stride;
    return {stride.data(), stride.size()};
}

/*******************************************************************************
 * Buffer requirement views
 *******************************************************************************/

vk::BufferUsageFlags BufferRequirementsView::usage() const {
    return detail::bufferMetadata(workloadResource(*workload_, resourceIndex_)).usage;
}

/*******************************************************************************
 * Image requirement views
 *******************************************************************************/

vk::Extent3D ImageRequirementsView::extent() const {
    return vulkan_helpers::imageExtentFromMetadata(workloadResource(*workload_, resourceIndex_));
}

vk::ImageUsageFlags ImageRequirementsView::usage() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    const auto usage = detail::imageMetadata(resource).usage;
    if (usage) {
        return usage;
    }
    if (!resource.descriptorType.has_value()) {
        return {};
    }
    return vulkan_helpers::imageUsage(*resource.descriptorType, resource.aliasGroupId.has_value());
}

bool ImageRequirementsView::isSampled() const {
    return workloadResource(*workload_, resourceIndex_).descriptorType == vk::DescriptorType::eCombinedImageSampler;
}

bool ImageRequirementsView::isStorage() const {
    return workloadResource(*workload_, resourceIndex_).descriptorType == vk::DescriptorType::eStorageImage;
}

bool ImageRequirementsView::hasRuntimeSampler() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    return resource.descriptorType == vk::DescriptorType::eCombinedImageSampler &&
           detail::imageMetadata(resource).samplerConfig.has_value();
}

bool ImageRequirementsView::requiresSamplerBinding() const { return isSampled() && !hasRuntimeSampler(); }

vk::ImageLayout ImageRequirementsView::requiredLayout() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    const auto layout = detail::imageMetadata(resource).layout;
    if (layout != vk::ImageLayout::eUndefined) {
        return layout;
    }
    return vulkan_helpers::imageLayout(*resource.descriptorType);
}

vk::ImageSubresourceRange ImageRequirementsView::requiredSubresourceRange() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    const auto subresourceRange = detail::imageMetadata(resource).subresourceRange;
    if (subresourceRange.aspectMask) {
        return subresourceRange;
    }
    return {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
}

/*******************************************************************************
 * Resource views
 *******************************************************************************/

uint32_t ResourceView::index() const { return index_; }

std::string_view ResourceView::name() const { return workloadResource(*workload_, resourceIndex_).name; }

ResourceAccess ResourceView::access() const { return workloadResourceAccess(*workload_, resourceIndex_); }

ResourceRequirementsView ResourceView::requirements() const { return {workload_, resourceIndex_}; }

/*******************************************************************************
 * Executable views
 *******************************************************************************/

uint32_t ExecutableView::index() const { return index_; }

std::string_view ExecutableView::name() const { return workloadExecutable(*workload_, index_).name; }

ExecutableKind ExecutableView::type() const { return workloadExecutable(*workload_, index_).type; }

ModuleView ExecutableView::module() const {
    const auto moduleIndex = workloadExecutable(*workload_, index_).moduleIndex;
    return {workload_, moduleIndex};
}

DataGraphPipelineMetadataView ExecutableView::dataGraphPipelineMetadata() const { return {workload_, index_}; }

/*******************************************************************************
 * Interface descriptor inspection
 *******************************************************************************/

uint32_t ExecutableView::interfaceDescriptorBindingCount() const {
    const auto &executable = workloadExecutable(*workload_, index_);
    uint32_t count = 0;
    for (const auto &descBinding : executable.bindings) {
        if (publicResourceIndexForInternalResourceIndex(*workload_, descBinding.resourceIndex).has_value()) {
            ++count;
        }
    }
    return count;
}

InterfaceDescriptorBindingInfo ExecutableView::interfaceDescriptorBinding(uint32_t bindingIndex) const {
    const auto &executable = workloadExecutable(*workload_, index_);
    uint32_t publicBindingIndex = 0;
    for (const auto &descBinding : executable.bindings) {
        const auto publicResourceIndex =
            publicResourceIndexForInternalResourceIndex(*workload_, descBinding.resourceIndex);
        if (!publicResourceIndex.has_value()) {
            continue;
        }
        if (publicBindingIndex++ != bindingIndex) {
            continue;
        }
        const auto descriptorType = workloadDescriptorType(*workload_, descBinding);
        return {descBinding.set,
                descBinding.binding,
                *publicResourceIndex,
                descBinding.access,
                vulkan_helpers::resourceKind(descriptorType),
                descriptorType};
    }
    throw std::out_of_range("Workload interface descriptor binding index out of range");
}

/*******************************************************************************
 * Data graph pipeline metadata views
 *******************************************************************************/

std::string_view DataGraphPipelineMetadataView::identifier() const {
    return workloadExecutable(*workload_, executableIndex_).dataGraphPipelineIdentifier;
}

vk::PipelineCreateFlags2 DataGraphPipelineMetadataView::flags() const {
    return workloadExecutable(*workload_, executableIndex_).dataGraphPipelineFlags;
}

ArrayView<const vk::SpecializationMapEntry> DataGraphPipelineMetadataView::specializationMapEntries() const {
    const auto &mapEntries = workloadExecutable(*workload_, executableIndex_).specializationInfo.mapEntries;
    return {mapEntries.data(), mapEntries.size()};
}

ArrayView<const uint8_t> DataGraphPipelineMetadataView::specializationData() const {
    const auto &data = workloadExecutable(*workload_, executableIndex_).specializationInfo.data;
    return {data.data(), data.size()};
}

/*******************************************************************************
 * Module views
 *******************************************************************************/

uint32_t ModuleView::index() const { return index_; }

std::string_view ModuleView::name() const { return workloadModule(*workload_, index_).name; }

std::string_view ModuleView::entryPoint() const { return workloadModule(*workload_, index_).entryPoint; }

ModuleCodeKind ModuleView::codeKind() const { return workloadModule(*workload_, index_).codeKind; }

bool ModuleView::requiresImplementation() const { return codeKind() == ModuleCodeKind::Missing; }

uint32_t PlaceholderModuleView::index() const { return index_; }

ModuleView PlaceholderModuleView::module() const { return {workload_, moduleIndex_}; }

} // namespace mlworkloadlib
