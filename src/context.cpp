/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal/context_impl.hpp"
#include "internal/utils.hpp"

#include <vulkan/vulkan_beta.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlworkloadlib {

namespace utils = detail::utils;
namespace vulkan_helpers = detail::vulkan_helpers;

namespace {

/*******************************************************************************
 * Internal helpers
 *******************************************************************************/

bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const char *name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
        return std::string_view(extension.extensionName.data()) == name;
    });
}

bool containsExtensionName(const std::vector<const char *> &extensions, const char *name) {
    return std::any_of(extensions.begin(), extensions.end(),
                       [name](const auto *extension) { return std::string_view(extension) == name; });
}

void appendDeviceExtension(std::vector<const char *> &extensions, const char *name) {
    if (name == nullptr) {
        throw std::runtime_error("Context::create() device extension name must not be null");
    }
    if (!containsExtensionName(extensions, name)) {
        extensions.push_back(name);
    }
}

bool hasExtensions(const std::vector<vk::ExtensionProperties> &availableExtensions,
                   const std::vector<const char *> &requiredExtensions) {
    return std::all_of(
        requiredExtensions.begin(), requiredExtensions.end(),
        [&availableExtensions](const auto *extension) { return hasExtension(availableExtensions, extension); });
}

uint32_t findExecutionQueueFamily(const vk::raii::PhysicalDevice &physicalDevice) {
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i) {
        const auto requiredFlags = vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eDataGraphARM;
        if ((queueFamilies[i].queueFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

std::vector<const char *> runtimeOwnedDeviceExtensions(const RuntimeContextDeviceRequirements &deviceRequirements) {
    std::vector<const char *> requiredDeviceExtensions = {
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_TENSORS_EXTENSION_NAME};
    for (const auto *extension : deviceRequirements.requiredDeviceExtensions) {
        appendDeviceExtension(requiredDeviceExtensions, extension);
    }
    return requiredDeviceExtensions;
}

struct RuntimeOwnedDeviceFeatures {
    RuntimeOwnedDeviceFeatures(const std::vector<vk::ExtensionProperties> &availableExtensions,
                               std::vector<const char *> requiredDeviceExtensions, void *deviceFeaturePNext)
        : deviceExtensions(std::move(requiredDeviceExtensions)), featureChain(&dataGraphFeatures) {
        deviceFeatures.shaderInt16 = true;
        deviceFeatures.shaderInt64 = true;

        vulkan12Features.storageBuffer8BitAccess = true;
        vulkan12Features.shaderInt8 = true;
        vulkan12Features.shaderFloat16 = true;
        vulkan12Features.vulkanMemoryModel = true;
        vulkan12Features.pNext = deviceFeaturePNext;

        vulkan13Features.synchronization2 = true;
        vulkan13Features.pipelineCreationCacheControl = true;
        vulkan13Features.pNext = &vulkan12Features;

        maintenance5Features.maintenance5 = true;
        maintenance5Features.pNext = &vulkan13Features;

        tensorFeatures.tensors = true;
        tensorFeatures.shaderTensorAccess = true;
        tensorFeatures.tensorNonPacked = true;
        tensorFeatures.pNext = &maintenance5Features;

        dataGraphFeatures.dataGraph = true;
        dataGraphFeatures.dataGraphShaderModule = true;
        dataGraphFeatures.dataGraphSpecializationConstants = true;
        dataGraphFeatures.pNext = &tensorFeatures;

        if (hasExtension(availableExtensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) {
            replicatedCompositesFeatures.shaderReplicatedComposites = true;
            replicatedCompositesFeatures.pNext = featureChain;
            featureChain = &replicatedCompositesFeatures;
            appendDeviceExtension(deviceExtensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        }
        // Enable extension if available, as it is required for some platforms (e.g. MoltenVK on Darwin).
        if (hasExtension(availableExtensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            appendDeviceExtension(deviceExtensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
    }

    RuntimeOwnedDeviceFeatures(const RuntimeOwnedDeviceFeatures &) = delete;
    RuntimeOwnedDeviceFeatures &operator=(const RuntimeOwnedDeviceFeatures &) = delete;
    RuntimeOwnedDeviceFeatures(RuntimeOwnedDeviceFeatures &&) = delete;
    RuntimeOwnedDeviceFeatures &operator=(RuntimeOwnedDeviceFeatures &&) = delete;
    ~RuntimeOwnedDeviceFeatures() = default;

    vk::PhysicalDeviceFeatures deviceFeatures;
    vk::PhysicalDeviceVulkan12Features vulkan12Features;
    vk::PhysicalDeviceVulkan13Features vulkan13Features;
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5Features;
    vk::PhysicalDeviceTensorFeaturesARM tensorFeatures;
    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeatures;
    vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures;
    std::vector<const char *> deviceExtensions;
    void *featureChain = nullptr;
};

vk::DeviceSize allocationSize(vk::DeviceSize memoryRequirementSize) {
    return memoryRequirementSize == 0 ? 1 : memoryRequirementSize;
}

} // namespace

/*******************************************************************************
 * Allocation state
 *******************************************************************************/

struct RuntimeAllocation::Impl {
    Impl() = default;
    virtual ~Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    Impl(Impl &&) = delete;
    Impl &operator=(Impl &&) = delete;

    vk::raii::DeviceMemory memory{nullptr};
    BoundMemoryInfo memoryInfo{};
};

/*******************************************************************************
 * Runtime allocation facade
 *******************************************************************************/

RuntimeAllocation::RuntimeAllocation(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

RuntimeAllocation::~RuntimeAllocation() = default;

RuntimeAllocation::RuntimeAllocation(RuntimeAllocation &&) noexcept = default;

RuntimeAllocation &RuntimeAllocation::operator=(RuntimeAllocation &&) noexcept = default;

RuntimeAllocation::Impl *RuntimeAllocation::runtimeAllocationImpl() noexcept { return impl_.get(); }

const RuntimeAllocation::Impl *RuntimeAllocation::runtimeAllocationImpl() const noexcept { return impl_.get(); }

BoundMemoryInfo RuntimeAllocation::memory() const {
    const auto *allocationState = this->runtimeAllocationImpl();
    return allocationState != nullptr ? allocationState->memoryInfo : BoundMemoryInfo{};
}

/*******************************************************************************
 * Tensor allocation facade
 *******************************************************************************/

struct TensorAllocation::Impl : RuntimeAllocation::Impl {
    vk::raii::TensorARM tensor{nullptr};
};

TensorAllocation::TensorAllocation() : RuntimeAllocation(std::make_unique<Impl>()) {}

TensorAllocation::Impl *TensorAllocation::tensorAllocationImpl() noexcept {
    return dynamic_cast<Impl *>(this->runtimeAllocationImpl());
}

const TensorAllocation::Impl *TensorAllocation::tensorAllocationImpl() const noexcept {
    return dynamic_cast<const Impl *>(this->runtimeAllocationImpl());
}

vk::TensorARM TensorAllocation::handle() const {
    const auto *allocationState = this->tensorAllocationImpl();
    return allocationState != nullptr ? *allocationState->tensor : vk::TensorARM(nullptr);
}

/*******************************************************************************
 * Buffer allocation facade
 *******************************************************************************/

struct BufferAllocation::Impl : RuntimeAllocation::Impl {
    vk::raii::Buffer buffer{nullptr};
};

BufferAllocation::BufferAllocation() : RuntimeAllocation(std::make_unique<Impl>()) {}

BufferAllocation::Impl *BufferAllocation::bufferAllocationImpl() noexcept {
    return dynamic_cast<Impl *>(this->runtimeAllocationImpl());
}

const BufferAllocation::Impl *BufferAllocation::bufferAllocationImpl() const noexcept {
    return dynamic_cast<const Impl *>(this->runtimeAllocationImpl());
}

vk::Buffer BufferAllocation::handle() const {
    const auto *allocationState = this->bufferAllocationImpl();
    return allocationState != nullptr ? *allocationState->buffer : vk::Buffer(nullptr);
}

/*******************************************************************************
 * Image allocation facade
 *******************************************************************************/

struct ImageAllocation::Impl : RuntimeAllocation::Impl {
    vk::raii::Image image{nullptr};
    vk::raii::ImageView imageView{nullptr};
    vk::ImageSubresourceRange subresourceRange;
};

ImageAllocation::ImageAllocation() : RuntimeAllocation(std::make_unique<Impl>()) {}

ImageAllocation::Impl *ImageAllocation::imageAllocationImpl() noexcept {
    return dynamic_cast<Impl *>(this->runtimeAllocationImpl());
}

const ImageAllocation::Impl *ImageAllocation::imageAllocationImpl() const noexcept {
    return dynamic_cast<const Impl *>(this->runtimeAllocationImpl());
}

vk::Image ImageAllocation::handle() const {
    const auto *allocationState = this->imageAllocationImpl();
    return allocationState != nullptr ? *allocationState->image : vk::Image(nullptr);
}

ImageBindingInfo ImageAllocation::binding() const {
    const auto *allocationState = this->imageAllocationImpl();
    if (allocationState == nullptr) {
        return {};
    }
    ImageBindingInfo binding;
    binding.image = *allocationState->image;
    binding.imageView = *allocationState->imageView;
    binding.layout = vk::ImageLayout::eUndefined;
    binding.subresourceRange = allocationState->subresourceRange;
    binding.memory = allocationState->memoryInfo;
    return binding;
}

/*******************************************************************************
 * Context creation and lifetime
 *******************************************************************************/

Context Context::create(const RuntimeContextDeviceRequirements &deviceRequirements) {
    auto owned = std::make_unique<Impl::Owned>();
    const auto requiredDeviceExtensions = runtimeOwnedDeviceExtensions(deviceRequirements);

    const vk::ApplicationInfo applicationInfo("mlworkloadlib", 1, nullptr, 0, VK_API_VERSION_1_3);
    std::vector<const char *> instanceExtensions;
    vk::InstanceCreateFlags instanceFlags;
    const auto availableInstanceExtensions = owned->raiiContext.enumerateInstanceExtensionProperties();
    if (hasExtension(availableInstanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instanceFlags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
    owned->instance =
        vk::raii::Instance(owned->raiiContext, vk::InstanceCreateInfo(instanceFlags, &applicationInfo, {}, {},
                                                                      static_cast<uint32_t>(instanceExtensions.size()),
                                                                      instanceExtensions.data()));

    std::string lastFailure;
    for (auto &candidate : vk::raii::PhysicalDevices(owned->instance)) {
        const auto extensions = candidate.enumerateDeviceExtensionProperties();
        if (!hasExtensions(extensions, requiredDeviceExtensions)) {
            continue;
        }

        const auto queueFamilyIndex = findExecutionQueueFamily(candidate);
        if (queueFamilyIndex == std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);
        RuntimeOwnedDeviceFeatures deviceFeatures(extensions, requiredDeviceExtensions,
                                                  deviceRequirements.deviceFeaturePNext);

        try {
            owned->physicalDevice = candidate;
            owned->queueFamilyIndex = queueFamilyIndex;
            owned->device = vk::raii::Device(candidate, {vk::DeviceCreateFlags(),
                                                         queueCreateInfo,
                                                         {},
                                                         deviceFeatures.deviceExtensions,
                                                         &deviceFeatures.deviceFeatures,
                                                         deviceFeatures.featureChain});
            owned->queue = owned->device.getQueue(queueFamilyIndex, 0);
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
    const auto requirements = resource.requirements();
    const auto resourceKind = requirements.kind();
    if (resourceKind != ResourceKind::Tensor) {
        throw std::runtime_error("Context::createTensor() requires a Tensor workload resource; actual kind is " +
                                 std::string(utils::resourceKindName(resourceKind)));
    }

    const auto tensorRequirements = requirements.asTensor();
    const auto shapeView = tensorRequirements.shape();
    const auto strideView = tensorRequirements.stride();
    std::vector<int64_t> shape(shapeView.begin(), shapeView.end());
    std::vector<int64_t> stride(strideView.begin(), strideView.end());
    if (shape.empty()) {
        throw std::runtime_error("Context::createTensor() requires non-empty tensor shape metadata");
    }

    auto usage = tensorRequirements.usage();
    if (!usage) {
        usage = vk::TensorUsageFlagBitsARM::eDataGraph;
    }
    const vk::TensorDescriptionARM description(vk::TensorTilingARM::eLinear, requirements.format(),
                                               static_cast<uint32_t>(shape.size()), shape.data(),
                                               stride.empty() ? nullptr : stride.data(), usage);
    const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);

    const auto contextView = this->contextView();
    auto tensor = vk::raii::TensorARM(contextView.device.get(), createInfo);
    const auto memoryRequirements = contextView.device.get()
                                        .getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*tensor))
                                        .memoryRequirements;
    const auto size = allocationSize(memoryRequirements.size);
    const auto memoryType = vulkan_helpers::findMemoryType(
        contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto memory = vk::raii::DeviceMemory(contextView.device.get(), vk::MemoryAllocateInfo(size, memoryType));
    contextView.device.get().bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*tensor, *memory, 0));

    TensorAllocation allocation;
    auto &allocationState = *allocation.tensorAllocationImpl();
    allocationState.memory = std::move(memory);
    allocationState.tensor = std::move(tensor);
    allocationState.memoryInfo = BoundMemoryInfo{*allocationState.memory, 0, size};
    return allocation;
}

BufferAllocation Context::createBuffer(ResourceView resource) const {
    const auto requirements = resource.requirements();
    const auto resourceKind = requirements.kind();
    if (resourceKind != ResourceKind::StorageBuffer) {
        throw std::runtime_error("Context::createBuffer() requires a StorageBuffer workload resource; actual kind is " +
                                 std::string(utils::resourceKindName(resourceKind)));
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
    const auto memoryType = vulkan_helpers::findMemoryType(
        contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto memory = vk::raii::DeviceMemory(contextView.device.get(),
                                         vk::MemoryAllocateInfo(allocationSize(memoryRequirements.size), memoryType));
    buffer.bindMemory(*memory, 0);

    BufferAllocation allocation;
    auto &allocationState = *allocation.bufferAllocationImpl();
    allocationState.memory = std::move(memory);
    allocationState.buffer = std::move(buffer);
    allocationState.memoryInfo = {*allocationState.memory, 0, allocationSize(memoryRequirements.size)};
    return allocation;
}

ImageAllocation Context::createImage(ResourceView resource) const {
    const auto requirements = resource.requirements();
    const auto resourceKind = requirements.kind();
    if (resourceKind != ResourceKind::Image) {
        throw std::runtime_error("Context::createImage() requires an Image workload resource; actual kind is " +
                                 std::string(utils::resourceKindName(resourceKind)));
    }

    const auto imageRequirements = requirements.asImage();
    const auto extent = imageRequirements.extent();
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
        throw std::runtime_error("Context::createImage() requires non-zero image extent metadata");
    }

    const auto imageType = extent.depth > 1 ? vk::ImageType::e3D : vk::ImageType::e2D;
    const auto viewType = extent.depth > 1 ? vk::ImageViewType::e3D : vk::ImageViewType::e2D;
    const auto usage = imageRequirements.usage();
    const auto contextView = this->contextView();
    auto image = vk::raii::Image(contextView.device.get(),
                                 vk::ImageCreateInfo({}, imageType, requirements.format(), extent, 1, 1,
                                                     vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage,
                                                     vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined));
    const auto memoryRequirements = image.getMemoryRequirements();
    const auto memoryType = vulkan_helpers::findMemoryType(
        contextView.physicalDevice.get(), memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto memory = vk::raii::DeviceMemory(contextView.device.get(),
                                         vk::MemoryAllocateInfo(allocationSize(memoryRequirements.size), memoryType));
    image.bindMemory(*memory, 0);

    const auto subresourceRange = imageRequirements.requiredSubresourceRange();
    auto imageView =
        vk::raii::ImageView(contextView.device.get(),
                            vk::ImageViewCreateInfo({}, *image, viewType, requirements.format(), {}, subresourceRange));

    ImageAllocation allocation;
    auto &allocationState = *allocation.imageAllocationImpl();
    allocationState.memory = std::move(memory);
    allocationState.image = std::move(image);
    allocationState.imageView = std::move(imageView);
    allocationState.memoryInfo = {*allocationState.memory, 0, allocationSize(memoryRequirements.size)};
    allocationState.subresourceRange = subresourceRange;
    return allocation;
}

} // namespace mlworkloadlib
