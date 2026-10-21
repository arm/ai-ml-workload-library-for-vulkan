/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "test_expected_results.hpp"
#include "test_resource_requirements.hpp"
#include "test_vulkan_fixture.hpp"

#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlsdk::workloadlib::test {

/*******************************************************************************
 * Alias execution harness
 *******************************************************************************/

enum class AliasScenario {
    OutputBufferAliasedToIntermediateBuffer,
    OutputBufferAliasedToIntermediateTensor,
    OutputTensorAliasedToIntermediateBuffer,
    OutputTensorAliasedToIntermediateTensor,
    OutputImageAliasedToIntermediateImage,
};

struct AliasCase {
    const char *name;
    AliasScenario scenario;
    uint32_t executableCount;
    uint32_t resourceCount;
};

inline constexpr std::array aliasCases = {
    AliasCase{"OutputBufferAliasedToIntermediateBuffer", AliasScenario::OutputBufferAliasedToIntermediateBuffer, 2, 5},
    AliasCase{"OutputBufferAliasedToIntermediateTensor", AliasScenario::OutputBufferAliasedToIntermediateTensor, 2, 4},
    AliasCase{"OutputTensorAliasedToIntermediateBuffer", AliasScenario::OutputTensorAliasedToIntermediateBuffer, 1, 3},
    AliasCase{"OutputTensorAliasedToIntermediateTensor", AliasScenario::OutputTensorAliasedToIntermediateTensor, 1, 2},
    AliasCase{"OutputImageAliasedToIntermediateImage", AliasScenario::OutputImageAliasedToIntermediateImage, 1, 4},
};

inline std::string aliasCaseName(const testing::TestParamInfo<AliasCase> &info) { return info.param.name; }

inline void PrintTo(const AliasCase &testCase, std::ostream *os) { *os << testCase.name; }

enum class AliasMemoryFailure { Undersized, Misaligned, Overflowing };

class AliasMemoryValidationMock {
  public:
    static constexpr vk::DeviceSize requiredSize = 64;
    static constexpr vk::DeviceSize requiredAlignment = 16;

    explicit AliasMemoryValidationMock(const vk::raii::Device &device) {
        const auto &dispatcher = *device.getDispatcher();
        ON_CALL(*this, GetBufferMemoryRequirements)
            .WillByDefault(testing::Invoke(dispatcher.vkGetBufferMemoryRequirements));
        ON_CALL(*this, GetTensorMemoryRequirements)
            .WillByDefault(testing::Invoke(dispatcher.vkGetTensorMemoryRequirementsARM));
        ON_CALL(*this, GetImageMemoryRequirements)
            .WillByDefault(testing::Invoke(dispatcher.vkGetImageMemoryRequirements));
        ON_CALL(*this, BindBufferMemory).WillByDefault(testing::Invoke(dispatcher.vkBindBufferMemory));
        ON_CALL(*this, BindTensorMemory).WillByDefault(testing::Invoke(dispatcher.vkBindTensorMemoryARM));
        ON_CALL(*this, BindImageMemory).WillByDefault(testing::Invoke(dispatcher.vkBindImageMemory));
    }

    ~AliasMemoryValidationMock() = default;

    AliasMemoryValidationMock(const AliasMemoryValidationMock &) = delete;
    AliasMemoryValidationMock &operator=(const AliasMemoryValidationMock &) = delete;
    AliasMemoryValidationMock(AliasMemoryValidationMock &&) = delete;
    AliasMemoryValidationMock &operator=(AliasMemoryValidationMock &&) = delete;

    MOCK_METHOD(void, GetBufferMemoryRequirements, (VkDevice, VkBuffer, VkMemoryRequirements *));
    MOCK_METHOD(void, GetTensorMemoryRequirements,
                (VkDevice, const VkTensorMemoryRequirementsInfoARM *, VkMemoryRequirements2 *));
    MOCK_METHOD(void, GetImageMemoryRequirements, (VkDevice, VkImage, VkMemoryRequirements *));
    MOCK_METHOD(VkResult, BindBufferMemory, (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize));
    MOCK_METHOD(VkResult, BindTensorMemory, (VkDevice, uint32_t, const VkBindTensorMemoryInfoARM *));
    MOCK_METHOD(VkResult, BindImageMemory, (VkDevice, VkImage, VkDeviceMemory, VkDeviceSize));
};

class AliasMemoryValidationHooks {
  public:
    class DeviceAccess {
      public:
        explicit DeviceAccess(vk::raii::Device &device) : DeviceAccess(device, std::defer_lock) { lock(); }

        DeviceAccess(vk::raii::Device &device, std::defer_lock_t)
            : device_(device), lock_(deviceAccessMutex_, std::defer_lock) {}

        ~DeviceAccess() {
            if (lock_.owns_lock()) {
                if (!hasAccess()) {
                    std::terminate();
                }
                --accessDepth_;
            }
        }

        DeviceAccess(const DeviceAccess &) = delete;
        DeviceAccess &operator=(const DeviceAccess &) = delete;
        DeviceAccess(DeviceAccess &&) = default;
        DeviceAccess &operator=(DeviceAccess &&) = delete;

        void lock() {
            lock_.lock();
            ++accessDepth_;
        }

        bool tryLock() {
            if (!lock_.try_lock()) {
                return false;
            }
            ++accessDepth_;
            return true;
        }

        vk::raii::Device &device() const {
            if (!lock_.owns_lock() || !hasAccess()) {
                throw std::logic_error("Alias device access requires the owning thread's lock");
            }
            return device_;
        }

        static bool hasAccess() { return accessDepth_ != 0; }

      private:
        vk::raii::Device &device_;
        std::unique_lock<std::recursive_mutex> lock_;
        inline static std::recursive_mutex deviceAccessMutex_;
        inline static thread_local uint32_t accessDepth_ = 0;
    };

    explicit AliasMemoryValidationHooks(DeviceAccess access)
        : deviceAccess_(std::move(access)), registration_(deviceAccess_.device()),
          getBufferRequirements_(deviceAccess_.device(),
                                 &vk::raii::detail::DeviceDispatcher::vkGetBufferMemoryRequirements,
                                 getBufferRequirements),
          getTensorRequirements_(deviceAccess_.device(),
                                 &vk::raii::detail::DeviceDispatcher::vkGetTensorMemoryRequirementsARM,
                                 getTensorRequirements),
          getImageRequirements_(deviceAccess_.device(),
                                &vk::raii::detail::DeviceDispatcher::vkGetImageMemoryRequirements,
                                getImageRequirements),
          bindBufferMemory_(deviceAccess_.device(), &vk::raii::detail::DeviceDispatcher::vkBindBufferMemory,
                            bindBufferMemory),
          bindTensorMemory_(deviceAccess_.device(), &vk::raii::detail::DeviceDispatcher::vkBindTensorMemoryARM,
                            bindTensorMemory),
          bindImageMemory_(deviceAccess_.device(), &vk::raii::detail::DeviceDispatcher::vkBindImageMemory,
                           bindImageMemory) {}

    ~AliasMemoryValidationHooks() {
        if (!DeviceAccess::hasAccess()) {
            std::terminate();
        }
    }

    AliasMemoryValidationHooks(const AliasMemoryValidationHooks &) = delete;
    AliasMemoryValidationHooks &operator=(const AliasMemoryValidationHooks &) = delete;
    AliasMemoryValidationHooks(AliasMemoryValidationHooks &&) = delete;
    AliasMemoryValidationHooks &operator=(AliasMemoryValidationHooks &&) = delete;

    std::shared_ptr<AliasMemoryValidationMock> mock() const { return active(registration_.device); }

  private:
    struct Registration {
        explicit Registration(const vk::raii::Device &deviceIn) : device(*deviceIn) {
            auto mock = std::make_shared<testing::NiceMock<AliasMemoryValidationMock>>(deviceIn);
            const auto inserted = active_.emplace(device, std::move(mock));
            if (!inserted.second) {
                throw std::logic_error("Alias memory hooks already installed for device");
            }
        }

        ~Registration() { active_.erase(device); }

        Registration(const Registration &) = delete;
        Registration &operator=(const Registration &) = delete;
        Registration(Registration &&) = delete;
        Registration &operator=(Registration &&) = delete;

        VkDevice device;
    };

    static std::shared_ptr<AliasMemoryValidationMock> active(VkDevice device) {
        if (!DeviceAccess::hasAccess()) {
            throw std::logic_error("Alias callbacks require guarded device access");
        }
        return active_.at(device);
    }

    static VKAPI_ATTR void VKAPI_CALL getBufferRequirements(VkDevice device, VkBuffer buffer,
                                                            VkMemoryRequirements *requirements) {
        active(device)->GetBufferMemoryRequirements(device, buffer, requirements);
    }

    static VKAPI_ATTR void VKAPI_CALL getTensorRequirements(VkDevice device,
                                                            const VkTensorMemoryRequirementsInfoARM *info,
                                                            VkMemoryRequirements2 *requirements) {
        active(device)->GetTensorMemoryRequirements(device, info, requirements);
    }

    static VKAPI_ATTR void VKAPI_CALL getImageRequirements(VkDevice device, VkImage image,
                                                           VkMemoryRequirements *requirements) {
        active(device)->GetImageMemoryRequirements(device, image, requirements);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL bindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                                           VkDeviceSize offset) {
        return active(device)->BindBufferMemory(device, buffer, memory, offset);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL bindTensorMemory(VkDevice device, uint32_t count,
                                                           const VkBindTensorMemoryInfoARM *info) {
        return active(device)->BindTensorMemory(device, count, info);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL bindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory,
                                                          VkDeviceSize offset) {
        return active(device)->BindImageMemory(device, image, memory, offset);
    }

    DeviceAccess deviceAccess_;
    Registration registration_;
    ScopedDeviceDispatcherHook<PFN_vkGetBufferMemoryRequirements> getBufferRequirements_;
    ScopedDeviceDispatcherHook<PFN_vkGetTensorMemoryRequirementsARM> getTensorRequirements_;
    ScopedDeviceDispatcherHook<PFN_vkGetImageMemoryRequirements> getImageRequirements_;
    ScopedDeviceDispatcherHook<PFN_vkBindBufferMemory> bindBufferMemory_;
    ScopedDeviceDispatcherHook<PFN_vkBindTensorMemoryARM> bindTensorMemory_;
    ScopedDeviceDispatcherHook<PFN_vkBindImageMemory> bindImageMemory_;
    inline static std::unordered_map<VkDevice, std::shared_ptr<AliasMemoryValidationMock>> active_;
};

class AliasExecutionTestBase : public RuntimeSessionExecutionTest {
  protected:
    void SetUp() override {
        RuntimeSessionExecutionTest::SetUp();
        if (!IsSkipped()) {
            memoryValidationHooks_.emplace(AliasMemoryValidationHooks::DeviceAccess(device));
        }
    }

    void expectInvalidAliasMemory(Workload &workload, const AliasCase &testCase, AliasMemoryFailure failure) {
        vk::DeviceSize offset = 0;
        vk::DeviceSize size = AliasMemoryValidationMock::requiredSize;
        const char *expectedMessage = nullptr;
        switch (failure) {
        case AliasMemoryFailure::Undersized:
            --size;
            expectedMessage = "Bound memory is too small for a resource";
            break;
        case AliasMemoryFailure::Misaligned:
            offset = 1;
            expectedMessage = "Bound memory offset does not satisfy resource alignment";
            break;
        case AliasMemoryFailure::Overflowing:
            offset = AliasMemoryValidationMock::requiredAlignment;
            size = std::numeric_limits<vk::DeviceSize>::max();
            expectedMessage = "Bound memory range overflows";
            break;
        }

        auto context = wrappedContext();
        std::vector<TensorAllocation> tensors;
        std::vector<BufferAllocation> buffers;
        std::vector<ImageAllocation> images;
        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);

        uint32_t aliasBindings = 0;
        const auto memoryForBinding = [&](ResourceView resource, BoundMemoryInfo memory) {
            if (resource.requirements().participatesInAliasing()) {
                ++aliasBindings;
                memory.offset = offset;
                memory.size = size;
            }
            return memory;
        };
        for (const auto resource : workload.resources()) {
            switch (resource.requirements().kind()) {
            case ResourceKind::Tensor: {
                const auto &tensor = tensors.emplace_back(context.createTensor(resource));
                bindings.bindTensor(resource, {tensor.handle(), memoryForBinding(resource, tensor.memory())});
                break;
            }
            case ResourceKind::StorageBuffer: {
                const auto &buffer = buffers.emplace_back(context.createBuffer(resource));
                bindings.bindBuffer(resource, {buffer.handle(), memoryForBinding(resource, buffer.memory())});
                break;
            }
            case ResourceKind::Image: {
                const auto &image = images.emplace_back(context.createImage(resource));
                auto binding = image.binding();
                binding.memory = memoryForBinding(resource, binding.memory);
                bindings.bindImage(resource, binding);
                break;
            }
            case ResourceKind::Unknown:
                FAIL() << "Unexpected resource kind in alias test";
            }
        }
        ASSERT_EQ(aliasBindings, 1);

        const auto mock = memoryValidationHooks_->mock();
        const VkMemoryRequirements requirements{AliasMemoryValidationMock::requiredSize,
                                                AliasMemoryValidationMock::requiredAlignment, 1};
        ON_CALL(*mock, GetBufferMemoryRequirements).WillByDefault(testing::SetArgPointee<2>(requirements));
        ON_CALL(*mock, GetTensorMemoryRequirements)
            .WillByDefault(testing::WithArg<2>(
                [requirements](VkMemoryRequirements2 *result) { result->memoryRequirements = requirements; }));
        ON_CALL(*mock, GetImageMemoryRequirements).WillByDefault(testing::SetArgPointee<2>(requirements));
        ON_CALL(*mock, BindBufferMemory).WillByDefault(testing::Return(VK_ERROR_INITIALIZATION_FAILED));
        ON_CALL(*mock, BindTensorMemory).WillByDefault(testing::Return(VK_ERROR_INITIALIZATION_FAILED));
        ON_CALL(*mock, BindImageMemory).WillByDefault(testing::Return(VK_ERROR_INITIALIZATION_FAILED));

        const bool intermediateBuffer = testCase.scenario == AliasScenario::OutputBufferAliasedToIntermediateBuffer ||
                                        testCase.scenario == AliasScenario::OutputTensorAliasedToIntermediateBuffer;
        const bool intermediateTensor = testCase.scenario == AliasScenario::OutputBufferAliasedToIntermediateTensor ||
                                        testCase.scenario == AliasScenario::OutputTensorAliasedToIntermediateTensor;
        const bool intermediateImage = testCase.scenario == AliasScenario::OutputImageAliasedToIntermediateImage;
        EXPECT_CALL(*mock, GetBufferMemoryRequirements).Times(intermediateBuffer ? 1 : 0);
        EXPECT_CALL(*mock, GetTensorMemoryRequirements).Times(intermediateTensor ? 1 : 0);
        EXPECT_CALL(*mock, GetImageMemoryRequirements).Times(intermediateImage ? 1 : 0);
        EXPECT_CALL(*mock, BindBufferMemory).Times(0);
        EXPECT_CALL(*mock, BindTensorMemory).Times(0);
        EXPECT_CALL(*mock, BindImageMemory).Times(0);

        try {
            (void)session.prepare(bindings);
            ADD_FAILURE() << "prepare() accepted invalid alias memory";
        } catch (const std::runtime_error &error) {
            EXPECT_STREQ(error.what(), expectedMessage);
        }
    }

    void runAliasCase(Workload &workload, const AliasCase &testCase, bool provideMemoryInfo,
                      std::optional<vk::TensorTilingARM> tensorTiling = std::nullopt,
                      std::optional<vk::ImageTiling> imageTiling = std::nullopt) {
        switch (testCase.scenario) {
        case AliasScenario::OutputBufferAliasedToIntermediateBuffer:
            runOutputBufferAliasedToIntermediateBuffer(workload, testCase, provideMemoryInfo);
            return;
        case AliasScenario::OutputBufferAliasedToIntermediateTensor:
            runOutputBufferAliasedToIntermediateTensor(workload, testCase, provideMemoryInfo);
            return;
        case AliasScenario::OutputTensorAliasedToIntermediateBuffer:
            runOutputTensorAliasedToIntermediateBuffer(workload, testCase, provideMemoryInfo);
            return;
        case AliasScenario::OutputTensorAliasedToIntermediateTensor:
            runOutputTensorAliasedToIntermediateTensor(workload, testCase, provideMemoryInfo, tensorTiling);
            return;
        case AliasScenario::OutputImageAliasedToIntermediateImage:
            runOutputImageAliasedToIntermediateImage(workload, testCase, provideMemoryInfo, imageTiling);
            return;
        }
        FAIL() << "Unhandled alias scenario";
    }

  private:
    std::optional<AliasMemoryValidationHooks> memoryValidationHooks_;

    void runOutputBufferAliasedToIntermediateBuffer(Workload &workload, const AliasCase &testCase,
                                                    bool provideMemoryInfo) {
        constexpr std::size_t elements = 10;
        constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);

        const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
        const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
        const Buffer zeroInputBuffer(physicalDevice, device, bufferSize);
        const Buffer aliasedOutputBuffer(physicalDevice, device, bufferSize);
        const Buffer finalOutputBuffer(physicalDevice, device, bufferSize);

        const std::vector<int32_t> firstInput = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
        const std::vector<int32_t> secondInput = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
        const std::vector<int32_t> zeros(elements, 0);

        firstInputBuffer.write(firstInput);
        secondInputBuffer.write(secondInput);
        zeroInputBuffer.write(zeros);
        aliasedOutputBuffer.write(zeros);
        finalOutputBuffer.write(zeros);

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);
        bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*zeroInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(3), BufferBindingInfo{*finalOutputBuffer.buffer});
        if (provideMemoryInfo) {
            bindings.bindBuffer(workload.resource(4),
                                BufferBindingInfo{*aliasedOutputBuffer.buffer,
                                                  {*aliasedOutputBuffer.memory, 0, aliasedOutputBuffer.memorySize}});
        } else {
            bindings.bindBuffer(workload.resource(4), BufferBindingInfo{*aliasedOutputBuffer.buffer});
        }

        if (!provideMemoryInfo) {
            EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
            return;
        }

        auto execution = session.prepare(bindings);
        execution.run();

        EXPECT_EQ(finalOutputBuffer.read(elements), addVectors(firstInput, secondInput));
        EXPECT_EQ(aliasedOutputBuffer.read(elements), addVectors(firstInput, secondInput));
    }

    void runOutputBufferAliasedToIntermediateTensor(Workload &workload, const AliasCase &testCase,
                                                    bool provideMemoryInfo) {
        constexpr std::size_t elements = 10;
        constexpr vk::DeviceSize aliasedBufferSize = 64 * sizeof(int32_t);
        constexpr vk::DeviceSize outputBufferSize = elements * sizeof(int32_t);

        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);

        const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
        const Buffer aliasedOutputBuffer(physicalDevice, device, aliasedBufferSize);
        const Buffer zeroInputBuffer(physicalDevice, device, outputBufferSize);
        const Buffer finalOutputBuffer(physicalDevice, device, outputBufferSize);

        const auto input = makeMaxpoolInput(inputTensor.shape, 5);
        const std::vector<int32_t> zeros(elements, 0);
        inputTensor.write(input);
        aliasedOutputBuffer.write(std::vector<int32_t>(64, 0));
        zeroInputBuffer.write(zeros);
        finalOutputBuffer.write(zeros);

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        if (provideMemoryInfo) {
            bindings.bindBuffer(workload.resource(1),
                                BufferBindingInfo{*aliasedOutputBuffer.buffer,
                                                  {*aliasedOutputBuffer.memory, 0, aliasedOutputBuffer.memorySize}});
        } else {
            bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*aliasedOutputBuffer.buffer});
        }
        bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*zeroInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(3), BufferBindingInfo{*finalOutputBuffer.buffer});

        if (!provideMemoryInfo) {
            EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
            return;
        }

        auto execution = session.prepare(bindings);
        execution.run();

        EXPECT_EQ(finalOutputBuffer.read(elements),
                  int32WordsFromBytes(expectedMaxpool(input, inputTensor.shape), elements));
    }

    void runOutputTensorAliasedToIntermediateBuffer(Workload &workload, const AliasCase &testCase,
                                                    bool provideMemoryInfo) {
        constexpr std::size_t elements = 10;
        constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);

        const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
        const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
        const Tensor aliasedOutputTensor(physicalDevice, device, vk::Format::eR8Sint,
                                         {static_cast<int64_t>(bufferSize)});

        const std::vector<int32_t> firstInput = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
        const std::vector<int32_t> secondInput = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
        firstInputBuffer.write(firstInput);
        secondInputBuffer.write(secondInput);
        aliasedOutputTensor.fill(0, static_cast<std::size_t>(bufferSize));

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);
        bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
        if (provideMemoryInfo) {
            bindings.bindTensor(workload.resource(2),
                                TensorBindingInfo{*aliasedOutputTensor.tensor,
                                                  {*aliasedOutputTensor.memory, 0, aliasedOutputTensor.memorySize}});
        } else {
            bindings.bindTensor(workload.resource(2), TensorBindingInfo{*aliasedOutputTensor.tensor});
        }

        if (!provideMemoryInfo) {
            EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
            return;
        }

        auto execution = session.prepare(bindings);
        execution.run();

        EXPECT_EQ(int32WordsFromBytes(aliasedOutputTensor.read(static_cast<std::size_t>(bufferSize)), elements),
                  addVectors(firstInput, secondInput));
    }

    void runOutputTensorAliasedToIntermediateTensor(Workload &workload, const AliasCase &testCase,
                                                    bool provideMemoryInfo, std::optional<vk::TensorTilingARM> tiling) {
        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);
        Session session(context, workload);
        if (tiling.has_value()) {
            session.setTensorTiling(workload.resource(1), *tiling);
        }

        const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
        const auto outputTiling = session.tensorTiling(workload.resource(1));
        const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16}, outputTiling,
                                  vk::TensorUsageFlagBitsARM::eDataGraph | vk::TensorUsageFlagBitsARM::eTransferSrc);

        const auto input = makeMaxpoolInput(inputTensor.shape, 5);
        inputTensor.write(input);
        if (outputTiling == vk::TensorTilingARM::eLinear) {
            outputTensor.fill(0, outputTensor.numElements());
        }

        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);
        bindings.bindTensor(workload.resource(0), TensorBindingInfo{*inputTensor.tensor});
        if (provideMemoryInfo) {
            bindings.bindTensor(
                workload.resource(1),
                TensorBindingInfo{*outputTensor.tensor, {*outputTensor.memory, 0, outputTensor.memorySize}});
        } else {
            bindings.bindTensor(workload.resource(1), TensorBindingInfo{*outputTensor.tensor});
        }

        if (!provideMemoryInfo) {
            EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
            return;
        }

        auto execution = session.prepare(bindings);
        execution.run();

        if (outputTiling == vk::TensorTilingARM::eOptimal) {
            // Read Optimal tensors through a Linear transfer destination rather
            // than assuming their device-memory representation is packed.
            const Tensor readbackTensor(
                physicalDevice, device, vk::Format::eR8Sint, outputTensor.shape, vk::TensorTilingARM::eLinear,
                vk::TensorUsageFlagBitsARM::eDataGraph | vk::TensorUsageFlagBitsARM::eTransferDst);
            recordAndSubmitCommands(device, queue, queueFamilyIndex, [&](vk::CommandBuffer commandBuffer) {
                const vk::MemoryBarrier2 beforeCopy(
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
                    vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferRead);
                commandBuffer.pipelineBarrier2(vk::DependencyInfo({}, beforeCopy), *device.getDispatcher());
                const vk::TensorCopyARM region;
                commandBuffer.copyTensorARM({*outputTensor.tensor, *readbackTensor.tensor, region},
                                            *device.getDispatcher());
                const vk::MemoryBarrier2 afterCopy(vk::PipelineStageFlagBits2::eAllTransfer,
                                                   vk::AccessFlagBits2::eTransferWrite,
                                                   vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
                commandBuffer.pipelineBarrier2(vk::DependencyInfo({}, afterCopy), *device.getDispatcher());
            });
            EXPECT_EQ(readbackTensor.read(readbackTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
        } else {
            EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
        }
    }

    void runOutputImageAliasedToIntermediateImage(Workload &workload, const AliasCase &testCase, bool provideMemoryInfo,
                                                  std::optional<vk::ImageTiling> tiling) {
        constexpr std::size_t elements = 10;
        constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);
        Session session(context, workload);
        if (tiling.has_value()) {
            session.setImageTiling(workload.resource(3), *tiling);
        }

        const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
        const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
        const Buffer outputBuffer(physicalDevice, device, bufferSize);
        const auto aliasedOutputImage =
            context.createImage(workload.resource(3), session.imageTiling(workload.resource(3)));

        const std::vector<int32_t> firstInput = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
        const std::vector<int32_t> secondInput = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
        firstInputBuffer.write(firstInput);
        secondInputBuffer.write(secondInput);
        outputBuffer.write(std::vector<int32_t>(elements, 0));

        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);
        bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

        auto imageBindingInfo = aliasedOutputImage.binding();
        // The public image supplies alias memory and is not dispatched itself.
        imageBindingInfo.layout.reset();
        if (!provideMemoryInfo) {
            imageBindingInfo.memory = {};
        }
        bindings.bindImage(workload.resource(3), imageBindingInfo);

        if (!provideMemoryInfo) {
            EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
            return;
        }

        auto execution = session.prepare(bindings);
        execution.run();

        EXPECT_EQ(outputBuffer.read(elements), addVectors(firstInput, secondInput));
    }
};

} // namespace mlsdk::workloadlib::test
