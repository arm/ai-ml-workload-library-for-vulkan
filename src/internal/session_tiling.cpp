/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "session_tiling.hpp"
#include "utils.hpp"
#include "workload_impl.hpp"

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
        if (peer.role != Resource::Role::Constant && peer.aliasGroupId == resource.aliasGroupId &&
            std::holds_alternative<Resource::ImageMetadata>(peer.metadata)) {
            usage |= vk::TensorUsageFlagBitsARM::eImageAliasing;
            break;
        }
    }
    return usage;
}

std::vector<ResourceTiling> defaultResourceTilings(const Workload &workload) {
    struct AliasGroupTiling {
        ResourceTiling tiling = ResourceTiling::Linear;
        bool requiresLinear = false;
        bool requiresOptimal = false;
        bool hasImage = false;
        bool hasOversizedTensor = false;
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
        if (image != nullptr) {
            group.hasImage = true;
        }
        if (tensor != nullptr && !resource.shape.empty() && resource.shape.back() > 4) {
            group.hasOversizedTensor = true;
        }
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
        // Both Linear and Optimal image aliases require the tensor's innermost
        // dimension to match the image format's component count.
        if (group.hasImage && group.hasOversizedTensor) {
            throw std::runtime_error("Alias group " + std::to_string(*resource.aliasGroupId) +
                                     " cannot alias an image with a tensor whose innermost dimension exceeds 4");
        }
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
