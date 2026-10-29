/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/module_compiler.hpp"
#include "internal/session_tiling.hpp"
#include "internal/utils.hpp"
#include "internal/workload_impl.hpp"

#include <vulkan/vulkan_format_traits.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mlsdk::workloadlib {

/*******************************************************************************
 * State access
 *******************************************************************************/

const Workload::Impl &Workload::workloadImpl() const noexcept { return *impl_; }

const Workload::Impl &workloadImpl(const Workload &workload) noexcept { return workload.workloadImpl(); }

using DescriptorBinding = detail::DescriptorBinding;
using Executable = detail::Executable;
using Module = detail::Module;
using Resource = detail::Resource;

/*******************************************************************************
 * Resource metadata
 *******************************************************************************/

detail::Resource::Role detail::Resource::publicRoleForAccess(ResourceAccess access) {
    switch (access) {
    case ResourceAccess::Read:
        return Resource::Role::Input;
    case ResourceAccess::Write:
    case ResourceAccess::ReadWrite:
        return Resource::Role::Output;
    }
    throw std::runtime_error("Unsupported workload resource access");
}

ResourceAccess detail::Resource::accessForRole(detail::Resource::Role role) {
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

vk::DeviceSize detail::Resource::byteSize() const { return resourceByteSize(*this); }

vk::Extent3D detail::Resource::imageExtent() const {
    if (const auto extent = imageExtentIfKnown()) {
        return *extent;
    }
    throw std::runtime_error("Workload image extent requires a concrete shape");
}

std::optional<vk::Extent3D> detail::Resource::imageExtentIfKnown() const {
    const auto &imageInfo = imageMetadata(*this);
    (void)detail::imageViewType(imageInfo.imageType);
    const auto validateExtent = [&imageInfo](vk::Extent3D extent) {
        if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
            throw std::runtime_error("Image dimensions must be positive");
        }
        if (imageInfo.imageType == vk::ImageType::e1D && (extent.height != 1 || extent.depth != 1)) {
            throw std::runtime_error("1D image height and depth must be 1");
        }
        if (imageInfo.imageType == vk::ImageType::e2D && extent.depth != 1) {
            throw std::runtime_error("2D image depth must be 1");
        }
        return extent;
    };
    if (imageInfo.extent.width != 0 || imageInfo.extent.height != 0 || imageInfo.extent.depth != 0) {
        return validateExtent(imageInfo.extent);
    }
    if (shape.empty()) {
        return std::nullopt;
    }

    if (shape.size() != 3 && shape.size() != 4) {
        throw std::runtime_error("Workload image shape must be HWC or NHWC; got rank " + std::to_string(shape.size()));
    }
    bool unresolved = false;
    for (const auto dimension : shape) {
        constexpr auto maxDimension = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        if (dimension != -1 && (dimension <= 0 || dimension > maxDimension)) {
            throw std::runtime_error("Workload image dimensions must be -1 or between 1 and UINT32_MAX; got " +
                                     std::to_string(dimension));
        }
        unresolved = unresolved || dimension == -1;
    }
    if (shape.size() == 4 && shape[0] != -1 && shape[0] != 1) {
        throw std::runtime_error("Workload NHWC image batch must be 1; got " + std::to_string(shape[0]));
    }
    const auto channelCount = vk::componentCount(format);
    if (!detail::isKnownFormat(format) || (shape.back() != -1 && shape.back() != channelCount)) {
        throw std::runtime_error("Workload image channel count must match format component count " +
                                 std::to_string(channelCount) + "; got " + std::to_string(shape.back()));
    }
    if (unresolved) {
        return std::nullopt;
    }
    const auto heightIndex = shape.size() - 3;
    return validateExtent(
        {static_cast<uint32_t>(shape[heightIndex + 1]), static_cast<uint32_t>(shape[heightIndex]), 1});
}

vk::ImageLayout detail::Resource::requiredImageLayout() const {
    const auto layout = imageMetadata(*this).layout;
    return layout != vk::ImageLayout::eUndefined ? layout : detail::imageLayout(*descriptorType);
}

vk::ImageSubresourceRange detail::Resource::requiredImageSubresourceRange() const {
    const auto &image = imageMetadata(*this);
    const auto formatAspects = detail::imageAspectMask(format);
    auto subresourceRange = image.subresourceRange;
    if (!subresourceRange.aspectMask) {
        const auto defaultAspects = descriptorType == vk::DescriptorType::eCombinedImageSampler &&
                                            (formatAspects & vk::ImageAspectFlagBits::eDepth)
                                        ? vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth}
                                        : formatAspects;
        subresourceRange = {defaultAspects, 0, 1, 0, 1};
    }
    const auto planeCount = vk::planeCount(format);
    const auto isValidPlaneAspect =
        (planeCount > 1 && (subresourceRange.aspectMask == vk::ImageAspectFlagBits::ePlane0 ||
                            subresourceRange.aspectMask == vk::ImageAspectFlagBits::ePlane1)) ||
        (planeCount > 2 && subresourceRange.aspectMask == vk::ImageAspectFlagBits::ePlane2);
    if ((subresourceRange.aspectMask & ~formatAspects) && !isValidPlaneAspect) {
        throw std::runtime_error("Image subresource aspects do not match its format");
    }
    if (descriptorType == vk::DescriptorType::eCombinedImageSampler &&
        subresourceRange.aspectMask == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        throw std::runtime_error("Sampled depth/stencil images must select one aspect");
    }
    return subresourceRange;
}

ResourceAccess Workload::Impl::resourceAccess(uint32_t resourceIndex) const {
    const auto &resource = resources.at(resourceIndex);
    std::optional<ResourceAccess> access;
    for (const auto &executable : executables) {
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
    return access.value_or(Resource::accessForRole(resource.role));
}

vk::DescriptorType Workload::Impl::descriptorTypeForBinding(const DescriptorBinding &descBinding) const {
    const auto &resource = resources.at(descBinding.resourceIndex);
    if (!resource.descriptorType.has_value()) {
        throw std::runtime_error("Descriptor binding references a workload resource without descriptor type");
    }
    return *resource.descriptorType;
}

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
    return resource.descriptorType ? detail::resourceKind(*resource.descriptorType) : ResourceKind::Unknown;
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
    if (resource.elementCount != 0) {
        return resource.elementCount;
    }
    if (!resource.shape.empty()) {
        return detail::elementCount(resource.shape);
    }
    if (std::holds_alternative<Resource::ImageMetadata>(resource.metadata) && detail::isKnownFormat(resource.format)) {
        if (const auto extent = resource.imageExtentIfKnown()) {
            return detail::elementCount(
                {extent->width, extent->height, extent->depth, vk::componentCount(resource.format)});
        }
    }
    return 0;
}

vk::DeviceSize ResourceRequirementsView::byteSize() const {
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    const auto *buffer = std::get_if<Resource::BufferMetadata>(&resource.metadata);
    if (buffer != nullptr && buffer->byteSize != 0) {
        return buffer->byteSize;
    }
    if (!detail::hasKnownFormatByteSize(resource.format)) {
        return 0;
    }
    if (std::holds_alternative<Resource::ImageMetadata>(resource.metadata)) {
        if (!resource.imageExtentIfKnown()) {
            return 0;
        }
    } else if (resource.shape.empty() && buffer != nullptr) {
        if (resource.elementCount == 0) {
            return 0;
        }
    } else if (resource.shape.empty() || std::any_of(resource.shape.begin(), resource.shape.end(),
                                                     [](int64_t dimension) { return dimension <= 0; })) {
        return 0;
    }
    return resource.byteSize();
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
                                 std::string(detail::resourceKindName(resourceKind)));
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

vk::TensorUsageFlagsARM TensorRequirementsView::usage(vk::TensorTilingARM tiling) const {
    return detail::tensorUsageForTiling(*workload_, resourceIndex_, tiling);
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

vk::ImageType ImageRequirementsView::imageType() const {
    return detail::imageMetadata(workloadResource(*workload_, resourceIndex_)).imageType;
}

vk::Extent3D ImageRequirementsView::extent() const {
    return workloadResource(*workload_, resourceIndex_).imageExtent();
}

vk::ImageUsageFlags ImageRequirementsView::usage() const { return usage(vk::ImageTiling::eOptimal); }

vk::ImageUsageFlags ImageRequirementsView::usage(vk::ImageTiling selectedTiling) const {
    if (selectedTiling != vk::ImageTiling::eLinear && selectedTiling != vk::ImageTiling::eOptimal) {
        throw std::runtime_error("Unsupported image tiling");
    }
    const auto &resource = workloadResource(*workload_, resourceIndex_);
    auto flags = detail::imageMetadata(resource).usage;
    if (resource.descriptorType.has_value()) {
        flags |= detail::imageUsage(*resource.descriptorType, resource.aliasGroupId.has_value());
    }
    return detail::imageUsageForTiling(flags, resource.aliasGroupId.has_value(), selectedTiling);
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
    return workloadResource(*workload_, resourceIndex_).requiredImageLayout();
}

vk::ImageSubresourceRange ImageRequirementsView::requiredSubresourceRange() const {
    return workloadResource(*workload_, resourceIndex_).requiredImageSubresourceRange();
}

/*******************************************************************************
 * Resource views
 *******************************************************************************/

uint32_t ResourceView::index() const { return index_; }

std::string_view ResourceView::name() const { return workloadResource(*workload_, resourceIndex_).name; }

ResourceAccess ResourceView::access() const { return workloadImpl(*workload_).resourceAccess(resourceIndex_); }

ResourceRequirementsView ResourceView::requirements() const { return {workload_, resourceIndex_}; }

uint32_t ResourceView::aliasPeerCount() const {
    const auto group = workloadResource(*workload_, resourceIndex_).aliasGroupId;
    if (!group.has_value()) {
        return 0;
    }
    const auto &resources = workloadResources(*workload_);
    return static_cast<uint32_t>(std::count_if(resources.begin(), resources.end(), [group](const auto &resource) {
        return resource.role == Resource::Role::Intermediate && resource.aliasGroupId == group;
    }));
}

AliasPeerView ResourceView::aliasPeer(uint32_t peerIndex) const {
    const auto group = workloadResource(*workload_, resourceIndex_).aliasGroupId;
    if (group.has_value()) {
        const auto &resources = workloadResources(*workload_);
        for (uint32_t index = 0; index < resources.size(); ++index) {
            if (resources[index].role == Resource::Role::Intermediate && resources[index].aliasGroupId == group) {
                if (peerIndex == 0) {
                    return {workload_, index};
                }
                --peerIndex;
            }
        }
    }
    throw std::out_of_range("Workload alias peer index out of range");
}

ResourceRequirementsView AliasPeerView::requirements() const { return {workload_, resourceIndex_}; }

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
        const auto descriptorType = workloadImpl(*workload_).descriptorTypeForBinding(descBinding);
        return {descBinding.set,
                descBinding.binding,
                *publicResourceIndex,
                descBinding.access,
                detail::resourceKind(descriptorType),
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

ArrayView<const std::byte> DataGraphPipelineMetadataView::specializationData() const {
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

} // namespace mlsdk::workloadlib
