/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/context_impl.hpp"
#include "internal/utils.hpp"

#include "mlworkloadlib/workload.hpp"

#include <vulkan/vulkan_beta.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlsdk::workloadlib {

namespace {

/*******************************************************************************
 * Internal helpers
 *******************************************************************************/

vk::DeviceSize allocationSize(vk::DeviceSize memoryRequirementSize) {
    return memoryRequirementSize == 0 ? 1 : memoryRequirementSize;
}

} // namespace

/*******************************************************************************
 * Allocation state
 *******************************************************************************/

struct RuntimeAllocation::Impl {
    struct ImageResource {
        vk::raii::Image image{nullptr};
        vk::raii::ImageView imageView{nullptr};
        vk::ImageSubresourceRange subresourceRange;
    };

    vk::raii::DeviceMemory memory{nullptr};
    BoundMemoryInfo memoryInfo{};
    // Resource handles are destroyed before their backing memory.
    std::variant<std::monostate, vk::raii::TensorARM, vk::raii::Buffer, ImageResource> resource;
};

/*******************************************************************************
 * Runtime allocation facade
 *******************************************************************************/

RuntimeAllocation::RuntimeAllocation() : impl_(std::make_unique<Impl>()) {}

RuntimeAllocation::~RuntimeAllocation() = default;

RuntimeAllocation::RuntimeAllocation(RuntimeAllocation &&) noexcept = default;

RuntimeAllocation &RuntimeAllocation::operator=(RuntimeAllocation &&) noexcept = default;

RuntimeAllocation::Impl *RuntimeAllocation::runtimeAllocationImpl() noexcept { return impl_.get(); }

const RuntimeAllocation::Impl *RuntimeAllocation::runtimeAllocationImpl() const noexcept { return impl_.get(); }

BoundMemoryInfo RuntimeAllocation::memory() const {
    const auto *allocationState = runtimeAllocationImpl();
    return allocationState != nullptr ? allocationState->memoryInfo : BoundMemoryInfo{};
}

/*******************************************************************************
 * Tensor allocation facade
 *******************************************************************************/

vk::TensorARM TensorAllocation::handle() const {
    const auto *allocationState = runtimeAllocationImpl();
    const auto *tensor =
        allocationState != nullptr ? std::get_if<vk::raii::TensorARM>(&allocationState->resource) : nullptr;
    return tensor != nullptr ? **tensor : vk::TensorARM(nullptr);
}

/*******************************************************************************
 * Buffer allocation facade
 *******************************************************************************/

vk::Buffer BufferAllocation::handle() const {
    const auto *allocationState = runtimeAllocationImpl();
    const auto *buffer =
        allocationState != nullptr ? std::get_if<vk::raii::Buffer>(&allocationState->resource) : nullptr;
    return buffer != nullptr ? **buffer : vk::Buffer(nullptr);
}

/*******************************************************************************
 * Image allocation facade
 *******************************************************************************/

vk::Image ImageAllocation::handle() const {
    const auto *allocationState = runtimeAllocationImpl();
    const auto *image =
        allocationState != nullptr ? std::get_if<Impl::ImageResource>(&allocationState->resource) : nullptr;
    return image != nullptr ? *image->image : vk::Image(nullptr);
}

ImageBindingInfo ImageAllocation::binding() const {
    const auto *allocationState = runtimeAllocationImpl();
    const auto *image =
        allocationState != nullptr ? std::get_if<Impl::ImageResource>(&allocationState->resource) : nullptr;
    if (image == nullptr) {
        return {};
    }
    ImageBindingInfo binding;
    binding.image = *image->image;
    binding.imageView = *image->imageView;
    binding.layout = vk::ImageLayout::eUndefined;
    binding.subresourceRange = image->subresourceRange;
    binding.memory = allocationState->memoryInfo;
    return binding;
}

/*******************************************************************************
 * Context creation and lifetime
 *******************************************************************************/

Context Context::create(const RuntimeContextDeviceRequirements &deviceRequirements) {
    auto owned = std::make_unique<Impl::Owned>();
    const auto requiredDeviceExtensions = detail::requiredWorkloadDeviceExtensions(deviceRequirements);
    const auto requiredQueueFlags = detail::requiredWorkloadQueueFlags(deviceRequirements);

    owned->instance = detail::createVulkanInstance(owned->raiiContext, "mlworkloadlib");

    std::string lastFailure;
    for (auto &candidate : vk::raii::PhysicalDevices(owned->instance)) {
        const auto extensions = candidate.enumerateDeviceExtensionProperties();
        const auto missingExtension =
            std::find_if(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end(),
                         [&extensions](const char *name) { return !detail::hasExtension(extensions, name); });
        if (missingExtension != requiredDeviceExtensions.end()) {
            lastFailure = std::string("Required Vulkan device extension is unavailable: ") + *missingExtension;
            continue;
        }

        const auto queueFamilyIndex =
            detail::findQueueFamilyIndex(candidate.getQueueFamilyProperties(), requiredQueueFlags);
        if (!queueFamilyIndex.has_value()) {
            lastFailure = "No nonempty Vulkan queue family supports " + vk::to_string(requiredQueueFlags);
            continue;
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, *queueFamilyIndex, 1, &queuePriority);

        try {
            const auto pipelineCreationCacheControl =
                candidate.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>()
                    .get<vk::PhysicalDeviceVulkan13Features>()
                    .pipelineCreationCacheControl;
            const detail::RuntimeDeviceConfiguration deviceConfiguration(
                extensions, requiredDeviceExtensions, deviceRequirements, pipelineCreationCacheControl);
            owned->physicalDevice = candidate;
            owned->queueFamilyIndex = *queueFamilyIndex;
            owned->device = vk::raii::Device(candidate, {vk::DeviceCreateFlags(),
                                                         queueCreateInfo,
                                                         {},
                                                         deviceConfiguration.extensions(),
                                                         &deviceConfiguration.features(),
                                                         deviceConfiguration.featureChain()});
            owned->queue = owned->device.getQueue(*queueFamilyIndex, 0);
            return Context(std::move(owned));
        } catch (const std::exception &error) {
            lastFailure = error.what();
        }
    }

    std::string message = "No Vulkan device supports the runtime-owned context requirements";
    if (!lastFailure.empty()) {
        message += ": " + lastFailure;
    }
    throw std::runtime_error(message);
}

Context Context::wrap(ContextView contextView) { return Context(std::make_unique<Impl::Wrapped>(contextView)); }

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Context::~Context() = default;

/*******************************************************************************
 * Context state access
 *******************************************************************************/

Context::Impl &Context::contextImpl() noexcept { return *impl_; }

const Context::Impl &Context::contextImpl() const noexcept { return *impl_; }

/*******************************************************************************
 * Context metadata
 *******************************************************************************/

ContextView Context::contextView() const { return contextImpl().contextView(); }

/*******************************************************************************
 * Context runtime-owned resource allocation
 *******************************************************************************/

TensorAllocation Context::createTensor(ResourceView resource) const {
    return createTensor(resource, vk::TensorTilingARM::eLinear);
}

TensorAllocation Context::createTensor(ResourceView resource, vk::TensorTilingARM tiling) const {
    const auto requirements = resource.requirements();
    const auto resourceKind = requirements.kind();
    if (resourceKind != ResourceKind::Tensor) {
        throw std::runtime_error("Context::createTensor() requires a Tensor workload resource; actual kind is " +
                                 std::string(detail::resourceKindName(resourceKind)));
    }

    const auto tensorRequirements = requirements.asTensor();
    const auto shapeView = tensorRequirements.shape();
    const auto strideView = tensorRequirements.stride();
    const auto usage = tensorRequirements.usage(tiling);
    std::vector<int64_t> shape(shapeView.begin(), shapeView.end());
    detail::validateTensorTiling(tiling, usage, shape, !strideView.empty());
    std::vector<int64_t> stride(strideView.begin(), strideView.end());
    detail::validateTensorMetadata(requirements.format(), shape, stride);

    vk::TensorDescriptionARM description;
    description.tiling = tiling;
    description.format = requirements.format();
    description.dimensionCount = static_cast<uint32_t>(shape.size());
    description.pDimensions = shape.data();
    description.pStrides = stride.empty() ? nullptr : stride.data();
    description.usage = usage;
    const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);

    const auto contextView = this->contextView();
    auto tensor = vk::raii::TensorARM(contextView.device.get(), createInfo);
    const auto memoryRequirements = contextView.device.get()
                                        .getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*tensor))
                                        .memoryRequirements;
    const auto size = allocationSize(memoryRequirements.size);
    const auto memoryType =
        detail::findMemoryType(contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto memory = vk::raii::DeviceMemory(contextView.device.get(), vk::MemoryAllocateInfo(size, memoryType));
    contextView.device.get().bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*tensor, *memory, 0));

    TensorAllocation allocation;
    auto &allocationState = *allocation.runtimeAllocationImpl();
    allocationState.memory = std::move(memory);
    allocationState.resource.emplace<vk::raii::TensorARM>(std::move(tensor));
    allocationState.memoryInfo = BoundMemoryInfo{*allocationState.memory, 0, size};
    return allocation;
}

BufferAllocation Context::createBuffer(ResourceView resource) const {
    const auto requirements = resource.requirements();
    const auto resourceKind = requirements.kind();
    if (resourceKind != ResourceKind::StorageBuffer) {
        throw std::runtime_error("Context::createBuffer() requires a StorageBuffer workload resource; actual kind is " +
                                 std::string(detail::resourceKindName(resourceKind)));
    }

    const auto size = requirements.byteSize();
    if (size == 0) {
        throw std::runtime_error("Context::createBuffer() requires non-zero buffer size metadata");
    }

    auto usage = requirements.asBuffer().usage();
    usage |= vk::BufferUsageFlagBits::eStorageBuffer;
    const auto contextView = this->contextView();
    auto buffer = vk::raii::Buffer(contextView.device.get(), vk::BufferCreateInfo({}, size, usage));
    const auto memoryRequirements = buffer.getMemoryRequirements();
    const auto memoryType =
        detail::findMemoryType(contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto memory = vk::raii::DeviceMemory(contextView.device.get(),
                                         vk::MemoryAllocateInfo(allocationSize(memoryRequirements.size), memoryType));
    buffer.bindMemory(*memory, 0);

    BufferAllocation allocation;
    auto &allocationState = *allocation.runtimeAllocationImpl();
    allocationState.memory = std::move(memory);
    allocationState.resource.emplace<vk::raii::Buffer>(std::move(buffer));
    allocationState.memoryInfo = {*allocationState.memory, 0, allocationSize(memoryRequirements.size)};
    return allocation;
}

ImageAllocation Context::createImage(ResourceView resource) const {
    return createImage(resource, vk::ImageTiling::eOptimal);
}

ImageAllocation Context::createImage(ResourceView resource, vk::ImageTiling tiling) const {
    const auto requirements = resource.requirements();
    const auto resourceKind = requirements.kind();
    if (resourceKind != ResourceKind::Image) {
        throw std::runtime_error("Context::createImage() requires an Image workload resource; actual kind is " +
                                 std::string(detail::resourceKindName(resourceKind)));
    }

    const auto imageRequirements = requirements.asImage();
    const auto usage = imageRequirements.usage(tiling);
    detail::validateImageTiling(tiling, usage);
    const auto extent = imageRequirements.extent();
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
        throw std::runtime_error("Context::createImage() requires non-zero image extent metadata");
    }

    const auto imageType = imageRequirements.imageType();
    const auto viewType = detail::imageViewType(imageType);

    const auto contextView = this->contextView();

    vk::ImageCreateInfo createInfo;
    if (requirements.participatesInAliasing()) {
        createInfo.flags |= vk::ImageCreateFlagBits::eAlias;
    }
    createInfo.imageType = imageType;
    createInfo.format = requirements.format();
    createInfo.extent = extent;
    createInfo.mipLevels = 1;
    createInfo.arrayLayers = 1;
    createInfo.samples = vk::SampleCountFlagBits::e1;
    createInfo.tiling = tiling;
    createInfo.usage = usage;
    createInfo.sharingMode = vk::SharingMode::eExclusive;
    createInfo.initialLayout = vk::ImageLayout::eUndefined;

    auto image = vk::raii::Image(contextView.device.get(), createInfo);

    const auto memoryRequirements = image.getMemoryRequirements();
    const auto memoryType = detail::findMemoryType(contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits,
                                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto memory = vk::raii::DeviceMemory(contextView.device.get(),
                                         vk::MemoryAllocateInfo(allocationSize(memoryRequirements.size), memoryType));
    image.bindMemory(*memory, 0);

    const auto subresourceRange = imageRequirements.requiredSubresourceRange();
    auto imageView =
        vk::raii::ImageView(contextView.device.get(),
                            vk::ImageViewCreateInfo({}, *image, viewType, requirements.format(), {}, subresourceRange));

    ImageAllocation allocation;
    auto &allocationState = *allocation.runtimeAllocationImpl();
    allocationState.memory = std::move(memory);
    allocationState.resource.emplace<RuntimeAllocation::Impl::ImageResource>(
        RuntimeAllocation::Impl::ImageResource{std::move(image), std::move(imageView), subresourceRange});
    allocationState.memoryInfo = {*allocationState.memory, 0, allocationSize(memoryRequirements.size)};
    return allocation;
}

} // namespace mlsdk::workloadlib
