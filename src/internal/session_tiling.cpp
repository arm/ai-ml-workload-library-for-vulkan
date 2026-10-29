/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "session_tiling.hpp"
#include "utils.hpp"
#include "workload_impl.hpp"

#include <vulkan/vulkan_format_traits.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>

namespace mlsdk::workloadlib::detail {

vk::TensorUsageFlagsARM tensorUsageForTiling(const Workload &workload, uint32_t resourceIndex,
                                             vk::TensorTilingARM tiling) {
    if (tiling != vk::TensorTilingARM::eLinear && tiling != vk::TensorTilingARM::eOptimal) {
        throw std::runtime_error("Unsupported tensor tiling");
    }

    const auto &resources = workloadImpl(workload).resources;
    const auto &resource = resources.at(resourceIndex);
    auto usage = tensorMetadata(resource).usage;
    if (!usage) {
        usage = vk::TensorUsageFlagBitsARM::eDataGraph;
    }
    if (tiling != vk::TensorTilingARM::eOptimal || resource.role == Resource::Role::Constant ||
        !resource.aliasGroupId.has_value()) {
        return usage;
    }

    // Optimal image peers use eTensorAliasingARM, so their tensor peers must
    // describe the same image-compatible layout.
    for (const auto &peer : resources) {
        if (peer.aliasGroupId == resource.aliasGroupId &&
            std::holds_alternative<Resource::ImageMetadata>(peer.metadata)) {
            usage |= vk::TensorUsageFlagBitsARM::eImageAliasing;
            break;
        }
    }
    return usage;
}

namespace {

// Check the declarations we own even though callers still guarantee that their
// external handles were created with those declarations.
void validateImageTensorAliases(const Workload &workload) {
    const auto &resources = workloadImpl(workload).resources;
    for (const auto &image : resources) {
        if (!image.aliasGroupId.has_value() || !std::holds_alternative<Resource::ImageMetadata>(image.metadata)) {
            continue;
        }
        const auto components = vk::componentCount(image.format);
        const auto imageType = imageMetadata(image).imageType;
        auto requiredRank = 4;
        if (imageType == vk::ImageType::e1D) {
            requiredRank = 2;
        } else if (imageType == vk::ImageType::e2D) {
            requiredRank = 3;
        }
        for (const auto &tensor : resources) {
            if (tensor.aliasGroupId != image.aliasGroupId || tensor.role == Resource::Role::Constant ||
                !std::holds_alternative<Resource::TensorMetadata>(tensor.metadata)) {
                continue;
            }
            // Image type determines dimensionality; extra outer tensor dimensions are singleton.
            if (tensor.shape.size() < static_cast<size_t>(requiredRank) || tensor.shape.back() != components ||
                !std::all_of(tensor.shape.begin(), tensor.shape.end() - requiredRank,
                             [](int64_t dimension) { return dimension == 1; })) {
                throw std::runtime_error("Image/tensor aliases require matching component count, sufficient rank and "
                                         "singleton outer dimensions");
            }
            const auto bits = vk::componentBits(tensor.format, 0);
            bool compatible = components != 0 && bits != 0 && vk::componentCount(tensor.format) == 1 &&
                              !vk::isCompressed(image.format) && vk::planeCount(image.format) == 1 &&
                              !vk::hasDepthComponent(image.format) && !vk::hasStencilComponent(image.format) &&
                              vk::blockSize(image.format) == components * vk::blockSize(tensor.format);
            for (uint8_t component = 0; component < components; ++component) {
                compatible = compatible && vk::componentBits(image.format, component) == bits;
            }
            if (!compatible) {
                throw std::runtime_error("Image/tensor alias component formats are incompatible or unsupported");
            }
        }
    }
}

} // namespace

std::vector<ResourceTiling> defaultResourceTilings(const Workload &workload) {
    validateImageTensorAliases(workload);

    struct AliasGroupTiling {
        ResourceTiling tiling = ResourceTiling::Linear;
        bool requiresLinear = false;
        bool requiresOptimal = false;
    };

    const auto &resources = workloadImpl(workload).resources;
    std::map<uint32_t, AliasGroupTiling> aliasGroupTilings;
    std::vector<ResourceTiling> resourceTilings;
    resourceTilings.reserve(resources.size());
    for (const auto &resource : resources) {
        const auto *tensor = std::get_if<Resource::TensorMetadata>(&resource.metadata);
        const auto *image = std::get_if<Resource::ImageMetadata>(&resource.metadata);
        const bool requiresOptimal = tensor != nullptr && resource.role != Resource::Role::Constant &&
                                     (tensor->usage & vk::TensorUsageFlagBitsARM::eImageAliasing);
        resourceTilings.emplace_back(image != nullptr || requiresOptimal ? ResourceTiling::Optimal
                                                                         : ResourceTiling::Linear);

        if (!resource.aliasGroupId.has_value() || resource.role == Resource::Role::Constant ||
            std::holds_alternative<std::monostate>(resource.metadata)) {
            continue;
        }

        auto &group = aliasGroupTilings[*resource.aliasGroupId];
        if (resourceTilings.back() == ResourceTiling::Optimal) {
            group.tiling = ResourceTiling::Optimal;
        }
        if (std::holds_alternative<Resource::BufferMetadata>(resource.metadata) ||
            (tensor != nullptr && !resource.stride.empty())) {
            group.requiresLinear = true;
        }
        if (tensor != nullptr && (tensor->usage & vk::TensorUsageFlagBitsARM::eImageAliasing)) {
            group.requiresOptimal = true;
            group.tiling = ResourceTiling::Optimal;
        }
    }

    // Resolve defaults for every alias peer before callers inspect or override tiling.
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const auto &resource = resources[index];
        if (!resource.aliasGroupId.has_value() || resource.role == Resource::Role::Constant ||
            std::holds_alternative<std::monostate>(resource.metadata)) {
            continue;
        }

        const auto &group = aliasGroupTilings.at(*resource.aliasGroupId);
        if (group.requiresLinear && group.requiresOptimal) {
            throw std::runtime_error("Alias group " + std::to_string(*resource.aliasGroupId) +
                                     " has incompatible tiling requirements");
        }
        resourceTilings[index] = group.requiresLinear ? ResourceTiling::Linear : group.tiling;
    }

    for (std::size_t index = 0; index < resources.size(); ++index) {
        const auto &resource = resources[index];
        if (resource.role == Resource::Role::Constant) {
            continue;
        }
        if (std::holds_alternative<Resource::TensorMetadata>(resource.metadata)) {
            const auto tiling = toTensorTiling(resourceTilings[index]);
            const auto usage = tensorUsageForTiling(workload, static_cast<uint32_t>(index), tiling);
            validateTensorTiling(tiling, usage, resource.shape, !resource.stride.empty());
        }
    }
    return resourceTilings;
}

void validateAliasGroupTilings(const Workload &workload, const std::vector<ResourceTiling> &resourceTilings) {
    std::map<uint32_t, ResourceTiling> groupTilings;
    const auto &resources = workloadImpl(workload).resources;
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const auto &resource = resources[index];
        if (!resource.aliasGroupId.has_value() || resource.role == Resource::Role::Constant ||
            std::holds_alternative<std::monostate>(resource.metadata)) {
            continue;
        }

        const auto tiling = resourceTilings.at(index);
        const auto [group, inserted] = groupTilings.emplace(*resource.aliasGroupId, tiling);
        if (!inserted && group->second != tiling) {
            throw std::runtime_error("Alias group " + std::to_string(*resource.aliasGroupId) +
                                     " has resources with different tiling");
        }
    }
}

void validateAliasImageTensorLayouts(const Workload &workload, const std::vector<ResourceTiling> &resourceTilings) {
    const auto &resources = workloadImpl(workload).resources;
    for (size_t imageIndex = 0; imageIndex < resources.size(); ++imageIndex) {
        const auto &image = resources[imageIndex];
        if (!image.aliasGroupId.has_value() || !std::holds_alternative<Resource::ImageMetadata>(image.metadata) ||
            resourceTilings.at(imageIndex) != ResourceTiling::Linear) {
            continue;
        }
        for (size_t tensorIndex = 0; tensorIndex < resources.size(); ++tensorIndex) {
            const auto &tensor = resources[tensorIndex];
            if (tensor.aliasGroupId != image.aliasGroupId || tensor.role == Resource::Role::Constant ||
                !std::holds_alternative<Resource::TensorMetadata>(tensor.metadata)) {
                continue;
            }
            const auto pair = "Alias group " + std::to_string(*image.aliasGroupId) + " image resource " +
                              std::to_string(imageIndex) + " and tensor resource " + std::to_string(tensorIndex);
            // We do not negotiate library-created image pitches/offsets before graph compilation.
            if (image.role == Resource::Role::Intermediate) {
                throw std::runtime_error(pair + ": linear image/tensor aliases require a caller-owned image");
            }
        }
    }
}

void selectResourceTiling(const Workload &workload, std::vector<ResourceTiling> &resourceTilings,
                          uint32_t resourceIndex, ResourceTiling tiling) {
    const auto &resources = workloadImpl(workload).resources;
    const auto &selectedResource = resources.at(resourceIndex);
    if (!selectedResource.aliasGroupId.has_value()) {
        resourceTilings.at(resourceIndex) = tiling;
        return;
    }

    std::vector<std::size_t> selections{resourceIndex};
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const auto &resource = resources[index];
        if (resource.role != Resource::Role::Intermediate || resource.aliasGroupId != selectedResource.aliasGroupId) {
            continue;
        }

        if (std::holds_alternative<Resource::TensorMetadata>(resource.metadata)) {
            const auto selectedTensorTiling = toTensorTiling(tiling);
            const auto usage = tensorUsageForTiling(workload, static_cast<uint32_t>(index), selectedTensorTiling);
            validateTensorTiling(selectedTensorTiling, usage, resource.shape, !resource.stride.empty());
            selections.emplace_back(index);
        } else if (const auto *image = std::get_if<Resource::ImageMetadata>(&resource.metadata)) {
            const auto selectedImageTiling = toImageTiling(tiling);
            const auto usage = image->usage | imageUsage(*resource.descriptorType, false);
            validateImageTiling(selectedImageTiling, imageUsageForTiling(usage, true, selectedImageTiling));
            selections.emplace_back(index);
        } else if (tiling == ResourceTiling::Optimal &&
                   std::holds_alternative<Resource::BufferMetadata>(resource.metadata)) {
            throw std::runtime_error("Optimal tiling cannot be used with a storage-buffer alias");
        }
    }

    // Validate every internal peer before committing any changes.
    for (const auto index : selections) {
        resourceTilings.at(index) = tiling;
    }
}

std::vector<std::optional<ResourceTiling>> fixedPipelineTilings(const Workload &workload,
                                                                const std::vector<ResourceTiling> &selectedTilings) {
    const auto &workloadState = workloadImpl(workload);
    const auto &resources = workloadState.resources;
    std::vector<std::optional<ResourceTiling>> tilings(selectedTilings.size());
    for (const auto &executable : workloadState.executables) {
        if (executable.type != ExecutableKind::Graph) {
            continue;
        }
        for (const auto &binding : executable.bindings) {
            if (workloadState.descriptorTypeForBinding(binding) != vk::DescriptorType::eTensorARM) {
                continue;
            }
            const auto resourceIndex = binding.resourceIndex;
            tilings.at(resourceIndex) = selectedTilings.at(resourceIndex);
            const auto &resource = resources.at(resourceIndex);
            if (!resource.aliasGroupId.has_value() || resource.role == Resource::Role::Constant) {
                continue;
            }
            for (std::size_t peerIndex = 0; peerIndex < resources.size(); ++peerIndex) {
                const auto &peer = resources[peerIndex];
                if (peer.aliasGroupId == resource.aliasGroupId && peer.role != Resource::Role::Constant &&
                    !std::holds_alternative<std::monostate>(peer.metadata)) {
                    tilings[peerIndex] = selectedTilings[peerIndex];
                }
            }
        }
    }
    return tilings;
}

void validateCompiledTilings(const std::vector<ResourceTiling> &selectedTilings,
                             const std::vector<std::optional<ResourceTiling>> &pipelineTilings) {
    for (std::size_t index = 0; index < pipelineTilings.size(); ++index) {
        if (pipelineTilings[index].has_value() && selectedTilings.at(index) != *pipelineTilings[index]) {
            throw std::runtime_error("Resource " + std::to_string(index) +
                                     " tiling differs from the compiled data graph pipeline");
        }
    }
}

} // namespace mlsdk::workloadlib::detail
