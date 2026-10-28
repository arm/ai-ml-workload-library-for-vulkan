/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/binding_types.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlsdk::workloadlib::detail {

// Validate the backing region and return the resource's allocation-relative offset.
inline vk::DeviceSize validateAliasMemoryBinding(uint32_t resourceIndex, const BoundMemoryInfo &memory,
                                                 vk::DeviceSize byteOffset, const vk::MemoryRequirements &requirements,
                                                 bool requiresDedicatedAllocation) {
    const auto fail = [resourceIndex](std::string_view message) {
        throw std::runtime_error("Alias resource " + std::to_string(resourceIndex) + ": " + std::string(message));
    };
    if (!memory.memory) {
        fail("missing external memory");
    }
    if (requiresDedicatedAllocation) {
        fail("a dedicated allocation is required; external alias backing cannot be used");
    }
    const auto maximum = std::numeric_limits<vk::DeviceSize>::max();
    if (memory.size == 0 || memory.size == vk::WholeSize || memory.size > maximum - memory.offset) {
        fail("supplied memory region is empty, unbounded, or overflows VkDeviceSize");
    }
    if (byteOffset >= memory.size) {
        fail("resource offset lies outside supplied memory region");
    }
    const auto effectiveOffset = memory.offset + byteOffset;
    if (requirements.alignment == 0 || effectiveOffset % requirements.alignment != 0) {
        fail("effective memory offset is misaligned");
    }
    if (requirements.size > memory.size - byteOffset) {
        fail("supplied memory region is too small for the resource's requirements");
    }
    if (requirements.memoryTypeBits == 0) {
        fail("resource has no compatible memory type");
    }
    return effectiveOffset;
}

} // namespace mlsdk::workloadlib::detail
