/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/alias_memory.hpp"
#include "internal/runtime_resource.hpp"
#include "internal/workload_builder.hpp"
#include "test_expected_results.hpp"
#include "test_resource_requirements.hpp"
#include "test_spirv_utils.hpp"
#include "test_vulkan_fixture.hpp"
#include "test_vulkan_resources.hpp"

#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib_utils/mapped_device_memory.hpp"

#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {
using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::test;

// A non-null identity for pure validation tests; never passed to Vulkan.
template <typename Handle> Handle handle() {
    if constexpr (std::is_pointer_v<Handle>) {
        static uint8_t identity = 0;
        return reinterpret_cast<Handle>(&identity);
    } else {
        return static_cast<Handle>(1);
    }
}

TEST(AliasMemory, AppliesRegionAndSubresourceOffsetExactlyOnce) { // cppcheck-suppress syntaxError
    const BoundMemoryInfo memory{handle<VkDeviceMemory>(), 1024, 1024};
    const vk::MemoryRequirements requirements{768, 256, 1};
    EXPECT_EQ(detail::validateAliasMemoryBinding(3, memory, 256, requirements, false), 1280);
    EXPECT_EQ(detail::validateAliasMemoryBinding(3, memory, 0, {1024, 256, 1}, false), 1024);
    EXPECT_EQ(detail::validateAliasMemoryBinding(3, {memory.memory, 1025, 1023}, 255, requirements, false), 1280);
}

TEST(AliasMemory, RejectsInvalidBindings) {
    const BoundMemoryInfo memory{handle<VkDeviceMemory>(), 1024, 1024};
    const vk::MemoryRequirements requirements{768, 256, 1};
    struct Case {
        const char *name;
        BoundMemoryInfo memory;
        vk::DeviceSize offset;
        vk::MemoryRequirements requirements;
        bool dedicated = false;
    };
    for (const auto &test : {Case{"misaligned", memory, 257, requirements},
                             {"insufficient remaining range", memory, 512, requirements},
                             {"offset outside region", memory, 1024, requirements},
                             {"dedicated", memory, 0, requirements, true},
                             {"empty region", {memory.memory, 0, 0}, 0, requirements},
                             {"unbounded region", {memory.memory, 0, vk::WholeSize}, 0, requirements},
                             {"overflow", {memory.memory, vk::WholeSize - 100, 200}, 0, requirements},
                             {"missing memory", {}, 0, requirements},
                             {"empty type mask", memory, 0, {1, 256, 0}},
                             {"zero alignment", memory, 0, {1, 0, 1}}}) {
        SCOPED_TRACE(test.name);
        EXPECT_THROW(detail::validateAliasMemoryBinding(3, test.memory, test.offset, test.requirements, test.dedicated),
                     std::runtime_error);
    }
}

TEST(AliasTensorDescription, ChecksDeviceLimitsForPaddedAndPackedTensors) {
    detail::Resource resource;
    resource.format = vk::Format::eR8Sint;
    resource.shape = {4, 4, 4};
    struct Case {
        const char *constraint;
        vk::PhysicalDeviceTensorPropertiesARM limits;
        bool packed = false;
    };
    for (const auto &test : {Case{"maxTensorDimensionCount 2", {2, 64, 4, 128, 512}},
                             {"maxPerDimensionTensorElements 3", {3, 64, 3, 128, 512}},
                             {"maxTensorElements 63", {3, 63, 4, 128, 512}},
                             {"maxTensorStride 127", {3, 64, 4, 127, 512}},
                             {"maxTensorSize 511", {3, 64, 4, 128, 511}},
                             {"maxTensorSize 63", {3, 64, 4, 128, 63}, true}}) {
        SCOPED_TRACE(::testing::Message() << test.constraint << ", packed=" << test.packed);
        const auto strides = test.packed ? std::vector<int64_t>{} : std::vector<int64_t>{128, 4, 1};
        EXPECT_THAT([&] { detail::validateAliasTensorDescription(3, resource, strides, &test.limits); },
                    ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
                        ::testing::HasSubstr("Alias tensor resource 3:"), ::testing::HasSubstr(test.constraint))));
    }
    const vk::PhysicalDeviceTensorPropertiesARM packedLimits{3, 64, 4, 128, 64};
    EXPECT_NO_THROW(detail::validateAliasTensorDescription(3, resource, {}, &packedLimits));
}

TEST(AliasTensorDescription, RejectsInvalidStrides) {
    detail::Resource resource;
    resource.format = vk::Format::eR8Sint;
    resource.shape = {4, 4, 4};
    struct Case {
        const char *name;
        std::vector<int64_t> strides;
        const char *reason;
    };
    for (const auto &test : {Case{"rank", {64, 16, 4, 1}, "stride count 4"},
                             {"zero", {16, 4, 0}, "dimension 2: byte stride 0"},
                             {"negative", {16, 4, -1}, "dimension 2: byte stride -1"},
                             {"overlap", {15, 4, 1}, "overlap adjacent dimensions"},
                             {"innermost stride", {32, 8, 2}, "innermost stride 2 must equal the element size 1"},
                             {"overflow", {std::numeric_limits<int64_t>::max(), 4, 1}, "overflows int64_t"}}) {
        SCOPED_TRACE(test.name);
        EXPECT_THAT([&] { detail::validateAliasTensorDescription(3, resource, test.strides); },
                    ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
                        ::testing::HasSubstr("Alias tensor resource 3:"), ::testing::HasSubstr(test.reason))));
    }
    resource.format = vk::Format::eR16Sint;
    EXPECT_THROW(detail::validateAliasTensorDescription(3, resource, {33, 8, 2}), std::runtime_error);
}

Workload makeTensorAliasWorkload(uint32_t peerCount = 1) {
    detail::WorkloadBuilder builder;
    using Role = detail::Resource::Role;
    const auto input =
        builder.addResource("input", tensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16}), Role::Input);
    builder.addResource("backing", makeBufferRequirements(4096), Role::Output, 17);
    ModuleImplementation module;
    module.codeKind = ModuleCodeKind::Spirv;
    module.spirv = assembleMaxpool8x8To4x4Spirv("alias", {0, 0, 0, 1});
    const auto moduleIndex = builder.addModule(std::move(module), "maxpool", "main");
    for (uint32_t i = 0; i < peerCount; ++i) {
        const auto tensor =
            builder.addResource("peer", tensorRequirements(vk::Format::eR8Sint, {1, 4, 4, 16}), Role::Intermediate, 17);
        const auto graph = builder.addExecutable("pool" + std::to_string(i), ExecutableKind::Graph, moduleIndex);
        builder.addDescriptorBinding(graph, input, 0, 0, ResourceAccess::Read);
        builder.addDescriptorBinding(graph, tensor, 0, 1, ResourceAccess::Write);
    }
    return builder.finish();
}

TEST(AliasLayout, RejectsIncompatibleImageTensorDescriptions) {
    for (const auto &tensor :
         {tensorRequirements(vk::Format::eR8Sint, {2, 2, 3}), tensorRequirements(vk::Format::eR8Sint, {2, 2}),
          tensorRequirements(vk::Format::eR8Sint, {2, 2, 2, 4}), tensorRequirements(vk::Format::eR32Sint, {2, 2, 4})}) {
        detail::WorkloadBuilder builder;
        builder.addResource("image", makeStorageImageRequirements(), detail::Resource::Role::Input, 17);
        builder.addResource("tensor", tensor, detail::Resource::Role::Intermediate, 17);
        const auto workload = builder.finish();
        EXPECT_THROW((void)detail::defaultResourceTilings(workload), std::runtime_error);
    }
}

BindingSet bindAliasResources(Session &session, const Workload &workload, const Tensor &input, const Buffer &backing) {
    auto bindings = session.createBindingSet();
    bindings.bindTensor(workload.resource(0), {*input.tensor});
    bindings.bindBuffer(workload.resource(1), {*backing.buffer, {*backing.memory, 0, backing.memorySize}});
    return bindings;
}

// Forward to Vulkan, recording tensor binds and optionally altering returned requirements.
class TensorMemoryCalls {
  public:
    explicit TensorMemoryCalls(vk::raii::Device &device)
        : get_(device, &vk::raii::detail::DeviceDispatcher::vkGetTensorMemoryRequirementsARM, get),
          bind_(device, &vk::raii::detail::DeviceDispatcher::vkBindTensorMemoryARM, bind) {
        active_ = this;
    }
    ~TensorMemoryCalls() { active_ = previous_; }
    TensorMemoryCalls(const TensorMemoryCalls &) = delete;
    TensorMemoryCalls &operator=(const TensorMemoryCalls &) = delete;
    TensorMemoryCalls(TensorMemoryCalls &&) = delete;
    TensorMemoryCalls &operator=(TensorMemoryCalls &&) = delete;

    std::function<void(VkMemoryRequirements2 &, size_t)> inject;
    mutable size_t queries = 0;
    mutable std::vector<vk::DeviceSize> offsets;

  private:
    static void VKAPI_CALL get(VkDevice device, const VkTensorMemoryRequirementsInfoARM *info,
                               VkMemoryRequirements2 *result) {
        const auto &self = *active_;
        self.get_.original()(device, info, result);
        if (self.inject) {
            self.inject(*result, self.queries);
        }
        ++self.queries;
    }
    static VkResult VKAPI_CALL bind(VkDevice device, uint32_t count, const VkBindTensorMemoryInfoARM *infos) {
        for (uint32_t i = 0; i < count; ++i) {
            active_->offsets.push_back(infos[i].memoryOffset);
        }
        return active_->bind_.original()(device, count, infos);
    }

    ScopedDeviceDispatcherHook<PFN_vkGetTensorMemoryRequirementsARM> get_;
    ScopedDeviceDispatcherHook<PFN_vkBindTensorMemoryARM> bind_;
    const TensorMemoryCalls *previous_ = active_;
    inline static thread_local const TensorMemoryCalls *active_ = nullptr;
};

class AliasMemoryExecutionTest : public RuntimeSessionExecutionTest {};

TEST_F(AliasMemoryExecutionTest, QueriesBufferAndImagePeerRequirements) {
    auto context = wrappedContext();
    for (const bool image : {false, true}) {
        SCOPED_TRACE(image);
        detail::WorkloadBuilder builder;
        const auto requirements = image ? makeStorageImageRequirements() : makeBufferRequirements(4096);
        builder.addResource("backing", requirements, detail::Resource::Role::Output, 17);
        builder.addResource("peer", requirements, detail::Resource::Role::Intermediate, 17);
        const auto workload = builder.finish();
        Session session(context, workload);
        const auto resource = workload.resource(0);
        EXPECT_THROW(session.setTensorStrides(resource.aliasPeer(0), {16, 4, 1}), std::runtime_error);
        session.configure();
        const auto queried = session.aliasPeerMemoryRequirements(resource.aliasPeer(0));
        if (image) {
            const auto allocation = context.createImage(resource, session.imageTiling(resource));
            EXPECT_EQ(queried.memory,
                      device.getImageMemoryRequirements2(vk::ImageMemoryRequirementsInfo2(allocation.handle()))
                          .memoryRequirements);
        } else {
            const auto allocation = context.createBuffer(resource);
            EXPECT_EQ(queried.memory,
                      device.getBufferMemoryRequirements2(vk::BufferMemoryRequirementsInfo2(allocation.handle()))
                          .memoryRequirements);
        }
    }
}

TEST_F(AliasMemoryExecutionTest, QueriesAndExecutesEachPeerWithItsOwnStridesAndPlacement) {
    auto workload = makeTensorAliasWorkload(2);
    auto context = wrappedContext();
    Session session(context, workload);
    const auto resource = workload.resource(1);
    ASSERT_EQ(workload.resourceCount(), 2u);
    ASSERT_EQ(resource.aliasPeerCount(), 2u);
    const std::array<std::vector<int64_t>, 2> strides{{{1024, 256, 16, 1}, {2048, 512, 16, 1}}};
    for (uint32_t i = 0; i < strides.size(); ++i) {
        session.setTensorStrides(resource.aliasPeer(i), strides[i]);
    }
    session.configure();
    std::array<vk::DeviceSize, 2> offsets{};
    vk::DeviceSize size = 4096;
    uint32_t memoryTypes = UINT32_MAX;
    for (uint32_t i = 0; i < strides.size(); ++i) {
        SCOPED_TRACE(i);
        const auto requirements = session.aliasPeerMemoryRequirements(resource.aliasPeer(i));
        EXPECT_FALSE(requirements.requiresDedicatedAllocation);
        const std::vector<int64_t> shape{1, 4, 4, 16};
        const vk::TensorDescriptionARM description(vk::TensorTilingARM::eLinear, vk::Format::eR8Sint, 4, shape.data(),
                                                   strides[i].data(), vk::TensorUsageFlagBitsARM::eDataGraph);
        const vk::raii::TensorARM probe(device, {{}, &description});
        EXPECT_EQ(
            requirements.memory,
            device.getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*probe)).memoryRequirements);
        const auto alignment = requirements.memory.alignment;
        offsets[i] = ((size + alignment - 1) / alignment) * alignment;
        size = offsets[i] + requirements.memory.size;
        memoryTypes &= requirements.memory.memoryTypeBits;
    }
    const Tensor input(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const auto values = makeMaxpoolInput(input.shape, 5);
    input.write(values);
    const Buffer backing(physicalDevice, device, size, memoryTypes);
    auto bindings = bindAliasResources(session, workload, input, backing);
    for (uint32_t i = 0; i < offsets.size(); ++i) {
        bindings.setAliasPeerOffset(resource.aliasPeer(i), offsets[i]);
    }
    const TensorMemoryCalls calls(device);
    auto execution = session.prepare(bindings);
    EXPECT_THAT(calls.offsets, ::testing::ElementsAre(offsets[0], offsets[1]));
    execution.run();
    const auto bytes =
        utils::readDeviceMemory<int8_t>(device, {*backing.memory, 0, backing.memorySize}, static_cast<size_t>(size));
    const auto expected = expectedMaxpool(values, input.shape);
    for (size_t i = 0; i < strides.size(); ++i) {
        SCOPED_TRACE(i);
        std::vector<int8_t> actual;
        for (size_t row = 0; row < 4; ++row) {
            const auto start = offsets[i] + (row * static_cast<size_t>(strides[i][1]));
            for (size_t column = 0; column < 64; ++column) {
                actual.push_back(bytes.at(static_cast<size_t>(start) + column));
            }
        }
        EXPECT_EQ(actual, expected);
    }
}

TEST_F(AliasMemoryExecutionTest, RejectsInvalidBackingAndCanRetryWithUpdatedBindings) {
    auto workload = makeTensorAliasWorkload();
    auto context = wrappedContext();
    Session session(context, workload);
    session.configure();
    const auto resource = workload.resource(1);
    const auto peer = resource.aliasPeer(0);
    const auto requirements = session.aliasPeerMemoryRequirements(peer).memory;
    const Tensor input(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Buffer backing(physicalDevice, device, std::max(vk::DeviceSize{4096}, requirements.size),
                         requirements.memoryTypeBits);
    auto bindings = bindAliasResources(session, workload, input, backing);
    const auto publicRequirements = backing.buffer.getMemoryRequirements();
    const TensorMemoryCalls calls(device);
    bindings.bindBuffer(resource, {*backing.buffer, {*backing.memory, 0, publicRequirements.size - 1}});
    EXPECT_THAT([&] { (void)session.prepare(bindings); },
                ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("region is too small")));
    if (publicRequirements.alignment > 1) {
        bindings.bindBuffer(resource, {*backing.buffer, {*backing.memory, 1, backing.memorySize - 1}});
        EXPECT_THAT([&] { (void)session.prepare(bindings); },
                    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("misaligned")));
    }
    bindings.bindBuffer(resource, {*backing.buffer, {*backing.memory, 0, backing.memorySize}});
    bindings.setAliasPeerOffset(peer, backing.memorySize);
    EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
    EXPECT_TRUE(calls.offsets.empty());
    bindings.setAliasPeerOffset(peer, 0);
    const auto first = session.prepare(bindings);

    const Buffer replacement(physicalDevice, device, backing.memorySize, requirements.memoryTypeBits);
    bindings.bindBuffer(resource, {*replacement.buffer, {*replacement.memory, 0, replacement.memorySize}});
    const auto second = session.prepare(bindings);
}

TEST_F(AliasMemoryExecutionTest, ChecksLayoutOverridesAcrossSharedSessions) {
    auto workload = makeTensorAliasWorkload();
    auto context = wrappedContext();
    CompiledExecution compiled(context, workload);
    Session first(context, workload, compiled);
    const auto peer = workload.resource(1).aliasPeer(0);
    EXPECT_THROW((void)first.aliasPeerMemoryRequirements(peer), std::runtime_error);
    first.setTensorStrides(peer, {1024, 256, 16, 1});
    EXPECT_THROW(first.setTensorStrides(peer, {15, 4, 1}), std::runtime_error);
    first.configure();
    EXPECT_THROW(first.setTensorStrides(peer, {2048, 512, 16, 1}), std::runtime_error);
    Session matching(context, workload, compiled);
    matching.setTensorStrides(peer, {1024, 256, 16, 1});
    EXPECT_NO_THROW(matching.configure());
    Session different(context, workload, compiled);
    EXPECT_THROW(different.configure(), std::runtime_error);
    EXPECT_THROW(different.setTensorStrides(peer, {2048, 512, 16, 1}), std::runtime_error);
    const auto other = makeTensorAliasWorkload();
    const auto foreign = other.resource(1).aliasPeer(0);
    EXPECT_THROW((void)first.aliasPeerMemoryRequirements(foreign), std::runtime_error);
    EXPECT_THROW(different.setTensorStrides(foreign, {1024, 256, 16, 1}), std::runtime_error);
    auto bindings = first.createBindingSet();
    EXPECT_THROW(bindings.setAliasPeerOffset(foreign, 0), std::runtime_error);
}

TEST_F(AliasMemoryExecutionTest, QueriesDedicatedHintsAndRejectsIncompatiblePeersBeforeAnyBind) {
    auto workload = makeTensorAliasWorkload(2);
    auto context = wrappedContext();
    Session session(context, workload);
    session.configure();
    const auto peer = workload.resource(1).aliasPeer(0);
    const auto requirements = session.aliasPeerMemoryRequirements(peer).memory;
    const Tensor input(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    const Buffer backing(physicalDevice, device, std::max(vk::DeviceSize{4096}, requirements.size),
                         requirements.memoryTypeBits);
    auto bindings = bindAliasResources(session, workload, input, backing);
    const auto typeIndex = detail::findMemoryType(
        physicalDevice, backing.buffer.getMemoryRequirements().memoryTypeBits & requirements.memoryTypeBits, {});
    {
        TensorMemoryCalls calls(device);
        calls.inject = [](VkMemoryRequirements2 &result, size_t) {
            auto *dedicated = static_cast<VkMemoryDedicatedRequirements *>(result.pNext);
            ASSERT_NE(dedicated, nullptr);
            dedicated->prefersDedicatedAllocation = VK_TRUE;
            dedicated->requiresDedicatedAllocation = VK_TRUE;
        };
        const auto hints = session.aliasPeerMemoryRequirements(peer);
        EXPECT_TRUE(hints.prefersDedicatedAllocation);
        EXPECT_TRUE(hints.requiresDedicatedAllocation);
        EXPECT_TRUE(calls.offsets.empty());
    }
    for (const auto *failure : {"dedicated", "no common memory type", "region is too small"}) {
        SCOPED_TRACE(failure);
        TensorMemoryCalls calls(device);
        calls.inject = [&](VkMemoryRequirements2 &result, size_t index) {
            if (std::string_view(failure) == "no common memory type") {
                const auto typeBit = uint32_t{1} << typeIndex;
                result.memoryRequirements.memoryTypeBits = index == 0 ? typeBit : ~typeBit;
                return;
            }
            if (index != 1) {
                return;
            }
            if (std::string_view(failure) == "dedicated") {
                static_cast<VkMemoryDedicatedRequirements *>(result.pNext)->requiresDedicatedAllocation = VK_TRUE;
            } else {
                result.memoryRequirements.size = backing.memorySize + 1;
            }
        };
        EXPECT_THAT([&] { (void)session.prepare(bindings); },
                    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(failure)));
        EXPECT_EQ(calls.queries, 2u);
        EXPECT_TRUE(calls.offsets.empty());
    }
}

TEST_F(AliasMemoryExecutionTest, RejectsUnsupportedLinearImageLayouts) {
    auto context = wrappedContext();
    for (const bool internalImage : {false, true}) {
        SCOPED_TRACE(internalImage);
        detail::WorkloadBuilder builder;
        using Role = detail::Resource::Role;
        builder.addResource("image", makeStorageImageRequirements(), internalImage ? Role::Intermediate : Role::Input,
                            17);
        const auto shape = internalImage ? std::vector<int64_t>{2, 2, 4} : std::vector<int64_t>{1, 2, 2, 4};
        auto tensor = tensorRequirements(vk::Format::eR8Sint, shape);
        if (internalImage) {
            tensor.tensor.stride = {8, 4, 1};
        }
        builder.addResource("tensor", tensor, Role::Intermediate, 17);
        const auto workload = builder.finish();
        Session session(context, workload);
        if (!internalImage) {
            session.setImageTiling(workload.resource(0), vk::ImageTiling::eLinear);
        }
        EXPECT_THAT([&] { session.configure(); }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
                                                      internalImage ? "caller-owned image" : "rank-three tensor")));
    }
}

} // namespace
