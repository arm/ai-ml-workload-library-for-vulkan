/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "byte_reader.hpp"
#include "workload_fuzzer.hpp"

#include "vgf/encoder.hpp"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

extern "C" const char *__asan_default_options() { return "detect_leaks=0"; }
extern "C" const char *__lsan_default_options() { return "detect_leaks=0"; }

namespace {

namespace workload_fuzz = mlworkloadlib::fuzz;
namespace vgflib = mlsdk::vgflib;

using workload_fuzz::ByteReader;

constexpr uint32_t kMaxResources = 5;
constexpr uint32_t kMaxModules = 3;
constexpr uint32_t kMaxSegments = 3;
constexpr uint32_t kMaxBindingSlots = 12;
constexpr uint32_t kMaxDescriptorSetsPerSegment = 3;
constexpr std::size_t kMaxGeneratedVgfSize = 64U * 1024U;
constexpr std::size_t kHeaderMutationWindow = 128;

enum class ResourceRole {
    Input,
    Output,
    Intermediate,
    Constant,
};

struct ResourceInfo {
    vgflib::ResourceRef ref;
    ResourceRole role;
    vgflib::DescriptorType descriptorType;
};

struct SlotInfo {
    vgflib::BindingSlotRef ref;
    ResourceRole role;
};

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "mlworkloadlib fuzzer invariant failed: %s\n", message);
    std::abort();
}

ResourceRole chooseRole(ByteReader &reader) {
    switch (reader.choose(4U)) {
    case 0:
        return ResourceRole::Input;
    case 1:
        return ResourceRole::Output;
    case 2:
        return ResourceRole::Intermediate;
    default:
        return ResourceRole::Constant;
    }
}

vgflib::DescriptorType chooseDescriptorType(ByteReader &reader) {
    constexpr std::array<vgflib::DescriptorType, 6> descriptorTypes = {
        static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
        static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_TENSOR_ARM),
        static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
        static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
        static_cast<vgflib::DescriptorType>(-1),
    };
    return descriptorTypes[reader.choose(static_cast<uint32_t>(descriptorTypes.size()))];
}

vgflib::FormatType chooseFormat(ByteReader &reader) {
    constexpr std::array<vgflib::FormatType, 8> formats = {
        static_cast<vgflib::FormatType>(VK_FORMAT_UNDEFINED),
        static_cast<vgflib::FormatType>(VK_FORMAT_R8_SINT),
        static_cast<vgflib::FormatType>(VK_FORMAT_R32_SINT),
        static_cast<vgflib::FormatType>(VK_FORMAT_R32_SFLOAT),
        static_cast<vgflib::FormatType>(VK_FORMAT_R8G8B8A8_SNORM),
        static_cast<vgflib::FormatType>(VK_FORMAT_R8G8B8A8_UNORM),
        static_cast<vgflib::FormatType>(VK_FORMAT_R4G4_UNORM_PACK8),
        static_cast<vgflib::FormatType>(VK_FORMAT_R4G4B4A4_UNORM_PACK16),
    };
    return formats[reader.choose(static_cast<uint32_t>(formats.size()))];
}

std::vector<int64_t> chooseShape(ByteReader &reader) {
    constexpr std::array<int64_t, 10> dimensions = {-2, -1, 0, 1, 2, 3, 4, 8, 16, 64};
    const auto rank = static_cast<std::size_t>(reader.choose(5U));
    std::vector<int64_t> shape;
    shape.reserve(rank);
    for (std::size_t dim = 0; dim < rank; ++dim) {
        shape.push_back(dimensions[reader.choose(static_cast<uint32_t>(dimensions.size()))]);
    }
    return shape;
}

std::vector<int64_t> chooseStrides(ByteReader &reader, std::size_t rank) {
    constexpr std::array<int64_t, 9> strides = {-16, -4, -1, 0, 1, 4, 8, 16, 64};
    std::size_t count = 0;
    switch (reader.choose(5U)) {
    case 0:
        count = 0;
        break;
    case 1:
        count = rank;
        break;
    case 2:
        count = rank == 0 ? 0 : rank - 1U;
        break;
    case 3:
        count = std::min<std::size_t>(rank + 1U, 5U);
        break;
    default:
        count = static_cast<std::size_t>(reader.choose(6U));
        break;
    }

    std::vector<int64_t> result;
    result.reserve(count);
    for (std::size_t strideIndex = 0; strideIndex < count; ++strideIndex) {
        result.push_back(strides[reader.choose(static_cast<uint32_t>(strides.size()))]);
    }
    return result;
}

std::optional<vgflib::AliasGroupId> chooseAliasGroup(ByteReader &reader, ResourceRole role) {
    if (role == ResourceRole::Constant || !reader.boolean()) {
        return std::nullopt;
    }
    return static_cast<vgflib::AliasGroupId>(1U + reader.choose(4U));
}

std::string chooseName(ByteReader &reader, std::string_view prefix) {
    constexpr uint32_t maxSuffixLength = 12;
    std::string name(prefix);
    const auto suffixLength = static_cast<std::size_t>(reader.choose(maxSuffixLength + 1U));
    for (std::size_t index = 0; index < suffixLength; ++index) {
        const int character = static_cast<int>('a') + static_cast<int>(reader.choose(26U));
        name.push_back(static_cast<char>(character));
    }
    return name;
}

std::vector<uint8_t> choosePayload(ByteReader &reader) {
    const auto size = static_cast<std::size_t>(1U + reader.choose(16U));
    std::vector<uint8_t> payload;
    payload.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
        payload.push_back(reader.consumeByte());
    }
    return payload;
}

std::vector<uint32_t> chooseSpirvCode(ByteReader &reader) {
    if (!reader.boolean()) {
        return {};
    }

    std::vector<uint32_t> code = {0x07230203U, 0x00010000U};
    const auto extraWords = reader.choose(4U);
    code.reserve(static_cast<std::size_t>(2U + extraWords));
    for (uint32_t index = 0; index < extraWords; ++index) {
        code.push_back(reader.consumeU32());
    }
    return code;
}

vgflib::ModuleType chooseModuleType(ByteReader &reader) {
    return reader.boolean() ? vgflib::ModuleType::COMPUTE : vgflib::ModuleType::GRAPH;
}

bool canUseAsInput(ResourceRole role) {
    return role == ResourceRole::Input || role == ResourceRole::Intermediate || role == ResourceRole::Constant;
}

bool canUseAsOutput(ResourceRole role) { return role == ResourceRole::Output || role == ResourceRole::Intermediate; }

std::vector<vgflib::BindingSlotRef> selectSlots(ByteReader &reader, const std::vector<SlotInfo> &slots,
                                                std::size_t maxCount) {
    std::vector<vgflib::BindingSlotRef> selected;
    selected.reserve(std::min(slots.size(), maxCount));
    for (const auto &slot : slots) {
        if (selected.size() >= maxCount) {
            break;
        }
        if (reader.boolean()) {
            selected.push_back(slot.ref);
        }
    }
    if (selected.empty() && !slots.empty() && reader.boolean()) {
        selected.push_back(slots[reader.choose(static_cast<uint32_t>(slots.size()))].ref);
    }
    return selected;
}

std::vector<vgflib::BindingSlotRef> selectSlotsForUse(ByteReader &reader, const std::vector<SlotInfo> &slots,
                                                      bool input) {
    std::vector<vgflib::BindingSlotRef> selected;
    selected.reserve(slots.size());
    for (const auto &slot : slots) {
        const bool usable = input ? canUseAsInput(slot.role) : canUseAsOutput(slot.role);
        if (usable && reader.boolean()) {
            selected.push_back(slot.ref);
        }
    }
    return selected;
}

std::vector<vgflib::GraphConstantBindingRef> selectConstantBindings(ByteReader &reader,
                                                                    const std::vector<vgflib::ConstantRef> &constants) {
    std::vector<vgflib::GraphConstantBindingRef> selected;
    selected.reserve(constants.size());
    for (const auto &constant : constants) {
        if (reader.boolean()) {
            selected.emplace_back(constant);
        }
    }
    return selected;
}

std::vector<std::string> chooseNames(ByteReader &reader, std::size_t count, const std::string_view &prefix) {
    if (!reader.boolean()) {
        return {};
    }

    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        names.push_back(chooseName(reader, prefix));
    }
    return names;
}

ResourceInfo addResource(vgflib::Encoder &encoder, ByteReader &reader) {
    const auto role = chooseRole(reader);
    const auto descriptorType = chooseDescriptorType(reader);
    const auto format = chooseFormat(reader);
    auto shape = chooseShape(reader);
    auto strides = chooseStrides(reader, shape.size());
    const auto aliasGroup = chooseAliasGroup(reader, role);

    switch (role) {
    case ResourceRole::Input:
        return {encoder.AddInputResource(descriptorType, format, shape, strides, aliasGroup), role, descriptorType};
    case ResourceRole::Output:
        return {encoder.AddOutputResource(descriptorType, format, shape, strides, aliasGroup), role, descriptorType};
    case ResourceRole::Intermediate:
        return {encoder.AddIntermediateResource(descriptorType, format, shape, strides, aliasGroup), role,
                descriptorType};
    case ResourceRole::Constant:
        return {encoder.AddConstantResource(format, shape, strides), role, descriptorType};
    }
    fail("unknown generated resource role");
}

void maybeAddSamplerConfig(vgflib::Encoder &encoder, ByteReader &reader, const ResourceInfo &resource) {
    const bool imageDescriptor =
        resource.descriptorType == static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ||
        resource.descriptorType == static_cast<vgflib::DescriptorType>(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    if (resource.role == ResourceRole::Constant || !imageDescriptor || !reader.boolean()) {
        return;
    }

    encoder.AddSamplerConfig(resource.ref, reader.choose(2U), reader.choose(2U), reader.choose(4U), reader.choose(4U),
                             reader.choose(5U));
}

std::vector<uint8_t> writeGeneratedVgf(ByteReader &reader) {
    auto encoder = vgflib::CreateEncoder(static_cast<uint16_t>(VK_HEADER_VERSION));

    std::vector<ResourceInfo> resources;
    const auto resourceCount = reader.choose(kMaxResources + 1U);
    resources.reserve(resourceCount);
    for (uint32_t index = 0; index < resourceCount; ++index) {
        resources.push_back(addResource(*encoder, reader));
        maybeAddSamplerConfig(*encoder, reader, resources.back());
    }

    std::vector<vgflib::ConstantRef> constants;
    constants.reserve(resources.size());
    for (const auto &resource : resources) {
        if (resource.role != ResourceRole::Constant || !reader.boolean()) {
            continue;
        }
        auto payload = choosePayload(reader);
        constants.push_back(encoder->AddConstant(resource.ref, payload.data(), payload.size(), -1));
    }

    std::vector<SlotInfo> slots;
    slots.reserve(std::min<uint32_t>(resourceCount * 2U, kMaxBindingSlots));
    for (const auto &resource : resources) {
        const auto slotCount = 1U + reader.choose(2U);
        for (uint32_t slotIndex = 0; slotIndex < slotCount && slots.size() < kMaxBindingSlots; ++slotIndex) {
            slots.push_back({encoder->AddBindingSlot(reader.choose(8U), resource.ref), resource.role});
        }
    }

    const auto segmentCount = reader.choose(kMaxSegments + 1U);
    const auto moduleCount = segmentCount == 0 ? reader.choose(kMaxModules + 1U) : 1U + reader.choose(kMaxModules);
    std::vector<vgflib::ModuleRef> modules;
    modules.reserve(moduleCount);
    for (uint32_t index = 0; index < moduleCount; ++index) {
        modules.push_back(encoder->AddModule(chooseModuleType(reader), chooseName(reader, "module"), "main",
                                             chooseSpirvCode(reader)));
    }

    for (uint32_t segmentIndex = 0; segmentIndex < segmentCount; ++segmentIndex) {
        std::vector<vgflib::DescriptorSetInfoRef> descriptorSets;
        const auto descriptorSetCount = slots.empty() ? 0 : reader.choose(kMaxDescriptorSetsPerSegment + 1U);
        descriptorSets.reserve(descriptorSetCount);
        for (uint32_t descriptorSetIndex = 0; descriptorSetIndex < descriptorSetCount; ++descriptorSetIndex) {
            const auto selectedSlots = selectSlots(reader, slots, 6U);
            const uint32_t setIndex = reader.choose(8U) == 0 ? std::numeric_limits<uint32_t>::max() : reader.choose(5U);
            descriptorSets.push_back(encoder->AddDescriptorSetInfo(selectedSlots, setIndex));
        }

        const std::array<uint32_t, 3> dispatchShape = {reader.choose(5U), reader.choose(5U), reader.choose(5U)};
        encoder->AddSegmentInfo(modules[reader.choose(static_cast<uint32_t>(modules.size()))],
                                chooseName(reader, "segment"), descriptorSets, selectSlotsForUse(reader, slots, true),
                                selectSlotsForUse(reader, slots, false), selectConstantBindings(reader, constants),
                                dispatchShape);
    }

    std::vector<vgflib::BindingSlotRef> modelInputs;
    std::vector<vgflib::BindingSlotRef> modelOutputs;
    for (const auto &slot : slots) {
        if (slot.role == ResourceRole::Input && reader.boolean()) {
            modelInputs.push_back(slot.ref);
        }
        if (slot.role == ResourceRole::Output && reader.boolean()) {
            modelOutputs.push_back(slot.ref);
        }
    }
    encoder->AddModelSequenceInputsOutputs(modelInputs, chooseNames(reader, modelInputs.size(), "input"), modelOutputs,
                                           chooseNames(reader, modelOutputs.size(), "output"));

    encoder->Finish();
    std::ostringstream stream(std::ios::binary);
    if (!encoder->WriteTo(stream)) {
        return {};
    }

    const auto encoded = stream.str();
    if (encoded.size() > kMaxGeneratedVgfSize) {
        return {};
    }

    std::vector<uint8_t> bytes;
    bytes.reserve(encoded.size());
    for (const char byte : encoded) {
        bytes.push_back(static_cast<uint8_t>(static_cast<unsigned char>(byte)));
    }
    return bytes;
}

void mutateBytes(ByteReader &reader, std::vector<uint8_t> &bytes) {
    if (bytes.empty()) {
        return;
    }

    const auto mutationCount = 1U + reader.choose(16U);
    for (uint32_t mutation = 0; mutation < mutationCount; ++mutation) {
        const auto window = reader.boolean() ? std::min(bytes.size(), kHeaderMutationWindow) : bytes.size();
        const auto index = static_cast<std::size_t>(reader.choose(static_cast<uint32_t>(window)));
        switch (reader.choose(5U)) {
        case 0:
            bytes[index] ^= reader.consumeByte(0xffU);
            break;
        case 1:
            bytes[index] = reader.consumeByte();
            break;
        case 2:
            bytes[index] = 0;
            break;
        case 3:
            bytes[index] = 0xffU;
            break;
        default:
            bytes[index] = static_cast<uint8_t>(static_cast<uint32_t>(bytes[index]) + reader.consumeByte(1U));
            break;
        }
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (data == nullptr || size == 0) {
        return 0;
    }

    ByteReader reader(data, size);
    const auto mode = reader.choose(8U);
    if (mode == 0) {
        workload_fuzz::fuzzWorkloadBytes(data, size);
        return 0;
    }

    auto bytes = writeGeneratedVgf(reader);
    if (bytes.empty()) {
        return 0;
    }
    if (mode >= 5U) {
        mutateBytes(reader, bytes);
    }
    workload_fuzz::fuzzWorkloadBytes(bytes.data(), bytes.size());
    return 0;
}
