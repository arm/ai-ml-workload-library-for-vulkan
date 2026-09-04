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

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

namespace mlworkloadlib::test {

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

inline std::string aliasCaseName(const testing::TestParamInfo<AliasCase> &info) { return info.param.name; }

inline void PrintTo(const AliasCase &testCase, std::ostream *os) { *os << testCase.name; }

class AliasExecutionTestBase : public RuntimeSessionExecutionTest {
  protected:
    void runAliasCase(Workload &workload, const AliasCase &testCase, bool provideMemoryInfo) {
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
            runOutputTensorAliasedToIntermediateTensor(workload, testCase, provideMemoryInfo);
            return;
        case AliasScenario::OutputImageAliasedToIntermediateImage:
            runOutputImageAliasedToIntermediateImage(workload, testCase, provideMemoryInfo);
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
                                                    bool provideMemoryInfo) {
        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);

        const Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
        const Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

        const auto input = makeMaxpoolInput(inputTensor.shape, 5);
        inputTensor.write(input);
        outputTensor.fill(0, outputTensor.numElements());

        Session session(context, workload);
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

        EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
    }

    void runOutputImageAliasedToIntermediateImage(Workload &workload, const AliasCase &testCase,
                                                  bool provideMemoryInfo) {
        constexpr std::size_t elements = 10;
        constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

        auto context = wrappedContext();
        ASSERT_EQ(workload.executableCount(), testCase.executableCount);

        const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
        const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
        const Buffer outputBuffer(physicalDevice, device, bufferSize);
        const Image aliasedOutputImage(physicalDevice, device, vk::ImageUsageFlagBits::eStorage);

        const std::vector<int32_t> firstInput = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
        const std::vector<int32_t> secondInput = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
        firstInputBuffer.write(firstInput);
        secondInputBuffer.write(secondInput);
        outputBuffer.write(std::vector<int32_t>(elements, 0));

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        ASSERT_EQ(workload.resourceCount(), testCase.resourceCount);
        bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
        bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

        const auto imageRequirements = workload.resource(3).requirements().asImage();
        ImageBindingInfo imageBindingInfo;
        imageBindingInfo.image = *aliasedOutputImage.image;
        imageBindingInfo.imageView = *aliasedOutputImage.imageView;
        imageBindingInfo.subresourceRange = imageRequirements.requiredSubresourceRange();
        if (provideMemoryInfo) {
            imageBindingInfo.memory = {*aliasedOutputImage.memory, 0, aliasedOutputImage.memorySize};
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

} // namespace mlworkloadlib::test
