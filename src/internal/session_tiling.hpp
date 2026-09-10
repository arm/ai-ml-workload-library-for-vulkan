/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace mlsdk::workloadlib {

class Workload;

namespace detail {

enum class ResourceTiling { Linear, Optimal };

constexpr ResourceTiling toResourceTiling(vk::TensorTilingARM tiling) {
    return tiling == vk::TensorTilingARM::eOptimal ? ResourceTiling::Optimal : ResourceTiling::Linear;
}

constexpr ResourceTiling toResourceTiling(vk::ImageTiling tiling) {
    return tiling == vk::ImageTiling::eOptimal ? ResourceTiling::Optimal : ResourceTiling::Linear;
}

constexpr vk::TensorTilingARM toTensorTiling(ResourceTiling tiling) {
    return tiling == ResourceTiling::Optimal ? vk::TensorTilingARM::eOptimal : vk::TensorTilingARM::eLinear;
}

constexpr vk::ImageTiling toImageTiling(ResourceTiling tiling) {
    return tiling == ResourceTiling::Optimal ? vk::ImageTiling::eOptimal : vk::ImageTiling::eLinear;
}

vk::TensorUsageFlagsARM tensorUsageForTiling(const Workload &workload, uint32_t resourceIndex,
                                             vk::TensorTilingARM tiling);

std::vector<ResourceTiling> defaultResourceTilings(const Workload &workload);
void validateAliasGroupTilings(const Workload &workload, const std::vector<ResourceTiling> &resourceTilings);
void validateAliasImageTensorLayouts(const Workload &workload, const std::vector<ResourceTiling> &resourceTilings);
void selectResourceTiling(const Workload &workload, std::vector<ResourceTiling> &resourceTilings,
                          uint32_t resourceIndex, ResourceTiling tiling);

std::vector<std::optional<ResourceTiling>> fixedPipelineTilings(const Workload &workload,
                                                                const std::vector<ResourceTiling> &selectedTilings);
void validateCompiledTilings(const std::vector<ResourceTiling> &selectedTilings,
                             const std::vector<std::optional<ResourceTiling>> &pipelineTilings);

} // namespace detail
} // namespace mlsdk::workloadlib
