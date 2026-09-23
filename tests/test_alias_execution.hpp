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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
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

class AliasExecutionTestBase : public RuntimeSessionExecutionTest {
  protected:
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
