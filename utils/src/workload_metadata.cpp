/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib_utils/workload_metadata.hpp"

#include "internal/utils.hpp"

#include <utility>

namespace mlsdk::workloadlib::utils {

std::string_view resourceKindName(ResourceKind kind) noexcept { return detail::resourceKindName(kind); }

std::string_view accessName(ResourceAccess access) noexcept {
    switch (access) {
    case ResourceAccess::Read:
        return "read";
    case ResourceAccess::Write:
        return "write";
    case ResourceAccess::ReadWrite:
        return "read/write";
    }
    return "unknown";
}

std::string_view executableKindName(ExecutableKind kind) noexcept {
    switch (kind) {
    case ExecutableKind::Graph:
        return "data graph";
    case ExecutableKind::Compute:
        return "compute";
    }
    return "unknown";
}

ResourceRequirements bufferRequirements(vk::DeviceSize byteSize) {
    ResourceRequirements requirements;
    requirements.kind = ResourceKind::StorageBuffer;
    requirements.descriptorType = vk::DescriptorType::eStorageBuffer;
    requirements.buffer.byteSize = byteSize;
    requirements.buffer.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    return requirements;
}

ResourceRequirements tensorRequirements(vk::Format format, std::vector<int64_t> shape) {
    ResourceRequirements requirements;
    requirements.kind = ResourceKind::Tensor;
    requirements.descriptorType = vk::DescriptorType::eTensorARM;
    requirements.format = format;
    requirements.elementCount = detail::elementCount(shape);
    requirements.tensor.shape = std::move(shape);
    requirements.tensor.usage = vk::TensorUsageFlagBitsARM::eDataGraph;
    return requirements;
}

} // namespace mlsdk::workloadlib::utils
