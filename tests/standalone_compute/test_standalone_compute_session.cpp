/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_standalone_compute_utils.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlworkloadlib;
using namespace mlworkloadlib::test;

class StandaloneComputeSessionExecutionTest : public RuntimeSessionExecutionTest {};

ComputeShaderDescription makeAddInt32BuffersWithSampledImagesDescription() {
    auto description = makeAddInt32BuffersDescription();
    description.resources.push_back(
        {"runtime_sampler_image", 0, 3, ResourceAccess::Read, makeSampledImageRequirements(true)});
    description.resources.push_back(
        {"caller_sampler_image", 0, 4, ResourceAccess::Read, makeSampledImageRequirements(false)});
    return description;
}

vk::raii::Sampler makeNearestSampler(const vk::raii::Device &device) {
    const vk::SamplerCreateInfo samplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                                                  vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
                                                  vk::SamplerAddressMode::eClampToEdge,
                                                  vk::SamplerAddressMode::eClampToEdge);
    return {device, samplerCreateInfo};
}

} // namespace

/*******************************************************************************
 * Positive coverage
 *******************************************************************************/

TEST_F(StandaloneComputeSessionExecutionTest, RunComputeShaderWorkload) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    const auto expected = addVectors(firstInput, secondInput);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, WrapUsesCallerOwnedVulkanObjects) {
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});
    const auto contextView = context.contextView();

    EXPECT_EQ(&contextView.instance.get(), &instance);
    EXPECT_EQ(&contextView.physicalDevice.get(), &physicalDevice);
    EXPECT_EQ(&contextView.device.get(), &device);
    EXPECT_EQ(&contextView.queue.get(), &queue);
    EXPECT_EQ(contextView.queueFamilyIndex, queueFamilyIndex);
}

TEST_F(StandaloneComputeSessionExecutionTest, RunGlslComputeShaderWorkload) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled in this build";
    }

    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeGlslAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    const std::vector<int32_t> secondInput = {10, -9, 8, -7, 6, -5, 4, -3, 2, -1};
    const auto expected = addVectors(firstInput, secondInput);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, RunHlslComputeShaderWorkload) {
    if (!supports(Feature::HlslModules)) {
        GTEST_SKIP() << "HLSL source module support is not enabled in this build";
    }

    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeHlslAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {2, -4, 6, -8, 10, -12, 14, -16, 18, -20};
    const std::vector<int32_t> secondInput = {1, 3, -5, -7, 9, 11, -13, -15, 17, 19};
    const auto expected = addVectors(firstInput, secondInput);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, RunComputeShaderWorkloadWithRuntimeCreatedBuffers) {
    constexpr size_t elements = 10;

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});

    auto firstInputBuffer = context.createBuffer(workload.resource(0));
    auto secondInputBuffer = context.createBuffer(workload.resource(1));
    auto outputBuffer = context.createBuffer(workload.resource(2));

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    const auto expected = addVectors(firstInput, secondInput);

    writeMappedMemory(device, firstInputBuffer.memory(), firstInput);
    writeMappedMemory(device, secondInputBuffer.memory(), secondInput);
    writeMappedMemory(device, outputBuffer.memory(), std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{firstInputBuffer.handle(), firstInputBuffer.memory()});
    bindings.bindBuffer(workload.resource(1),
                        BufferBindingInfo{secondInputBuffer.handle(), secondInputBuffer.memory()});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{outputBuffer.handle(), outputBuffer.memory()});

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(readMappedMemory<int32_t>(device, outputBuffer.memory(), elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, MovesRuntimeOwnedBufferAndImageAllocations) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersWithSampledImagesDescription());
    auto context = wrappedContext();

    auto buffer = context.createBuffer(workload.resource(0));
    auto movedBuffer(std::move(buffer));
    auto assignedBuffer = context.createBuffer(workload.resource(1));
    EXPECT_NE(assignedBuffer.handle(), nullptr);
    assignedBuffer = std::move(movedBuffer);

    EXPECT_NE(assignedBuffer.handle(), nullptr);
    EXPECT_NE(assignedBuffer.memory().memory, nullptr);

    auto image = context.createImage(workload.resource(3));
    auto movedImage(std::move(image));
    auto assignedImage = context.createImage(workload.resource(4));
    EXPECT_NE(assignedImage.handle(), nullptr);
    assignedImage = std::move(movedImage);

    EXPECT_NE(assignedImage.handle(), nullptr);
    EXPECT_NE(assignedImage.memory().memory, nullptr);

    const auto imageBinding = assignedImage.binding();
    EXPECT_NE(imageBinding.image, nullptr);
    EXPECT_NE(imageBinding.imageView, nullptr);
    EXPECT_NE(imageBinding.memory.memory, nullptr);
    ASSERT_TRUE(imageBinding.layout.has_value());
    EXPECT_EQ(*imageBinding.layout, vk::ImageLayout::eUndefined);
}

TEST_F(StandaloneComputeSessionExecutionTest, RunGlslComputeShaderWithPushAndSpecializationConstants) {
    if (!supports(Feature::GlslModules)) {
        GTEST_SKIP() << "GLSL source module support is not enabled in this build";
    }

    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);
    constexpr int32_t pushAdd = 7;
    constexpr int32_t specializationAdd = 11;
    constexpr int32_t buildOptionAdd = 3;
    constexpr int32_t includeAdd = 5;

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersGlslDescription(specializationAdd));
    auto context = Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue});

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    const std::vector<int32_t> secondInput = {-1, -2, 3, 4, -5, 6, 7, -8, 9, 10};
    auto expected = addVectors(firstInput, secondInput);
    for (auto &value : expected) {
        value += pushAdd + specializationAdd + buildOptionAdd + includeAdd;
    }

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});
    bindings.bindPushConstants(&pushAdd, sizeof(pushAdd));

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, RunComputeShaderRepeatedDifferentInput) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    for (int32_t run = 0; run < 5; ++run) {
        std::vector<int32_t> firstInput(elements);
        std::vector<int32_t> secondInput(elements);
        for (size_t element = 0; element < elements; ++element) {
            firstInput[element] = (run + 1) * static_cast<int32_t>(element + 1);
            secondInput[element] = run * 3 - static_cast<int32_t>(element);
        }

        firstInputBuffer.write(firstInput);
        secondInputBuffer.write(secondInput);
        outputBuffer.write(std::vector<int32_t>(elements, 0));
        execution.run();
        EXPECT_EQ(outputBuffer.read(elements), addVectors(firstInput, secondInput));
    }
}

TEST_F(StandaloneComputeSessionExecutionTest, RunComputeShaderWorkloadWithExternalImageBinding) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto description = makeAddInt32BuffersDescription();
    description.resources.push_back({"image", 0, 3, ResourceAccess::Read, makeSampledImageRequirements(false)});
    auto workload = Workload::fromComputeShader(std::move(description));
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);
    const Image imageInput(physicalDevice, device);

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    const auto expected = addVectors(firstInput, secondInput);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 4);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    const vk::SamplerCreateInfo samplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                                                  vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
                                                  vk::SamplerAddressMode::eClampToEdge,
                                                  vk::SamplerAddressMode::eClampToEdge);
    const vk::raii::Sampler sampler(device, samplerCreateInfo);
    const auto imageRequirements = workload.resource(3).requirements().asImage();
    ImageBindingInfo imageBindingInfo;
    imageBindingInfo.image = *imageInput.image;
    imageBindingInfo.imageView = *imageInput.imageView;
    imageBindingInfo.sampler = *sampler;
    imageBindingInfo.subresourceRange = imageRequirements.requiredSubresourceRange();
    bindings.bindImage(workload.resource(3), imageBindingInfo);

    auto execution = session.prepare(bindings);
    execution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, RecordComputeShaderWorkload) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {2, 4, 6, 8, 10, -12, -14, 16, 18, 20};
    const std::vector<int32_t> secondInput = {3, -3, 5, -5, 7, -7, 11, -11, 13, -13};
    const auto expected = addVectors(firstInput, secondInput);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    ASSERT_EQ(workload.resourceCount(), 3);
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto execution = session.prepare(bindings);
    const vk::raii::CommandPool commandPool(device,
                                            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    auto commandBuffer =
        std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    execution.record(*commandBuffer);
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    ASSERT_EQ(device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()), vk::Result::eSuccess);

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(StandaloneComputeSessionExecutionTest, RunComputeShaderWithMovedBindingSetAndPreparedExecution) {
    constexpr std::size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    const Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {5, 8, 13, 21, 34, -55, -89, 144, 233, 377};
    const std::vector<int32_t> secondInput = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    const auto expected = addVectors(firstInput, secondInput);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(2), BufferBindingInfo{*outputBuffer.buffer});

    auto movedBindings(std::move(bindings));
    auto assignedBindings = std::move(movedBindings);

    auto execution = session.prepare(assignedBindings);
    auto movedExecution(std::move(execution));
    auto assignedExecution = std::move(movedExecution);
    assignedExecution.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

/*******************************************************************************
 * Validation coverage
 *******************************************************************************/

TEST_F(StandaloneComputeSessionExecutionTest, RejectsRuntimeAllocationForWrongResourceKind) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersWithSampledImagesDescription());
    auto context = wrappedContext();

    EXPECT_THROW((void)context.createTensor(workload.resource(0)), std::runtime_error);
    EXPECT_THROW((void)context.createBuffer(workload.resource(3)), std::runtime_error);
    EXPECT_THROW((void)context.createImage(workload.resource(0)), std::runtime_error);
}

TEST_F(StandaloneComputeSessionExecutionTest, ConfigureRejectsDuplicateDescriptorBindings) {
    auto description = makeAddInt32BuffersDescription();
    description.resources[1].set = description.resources[0].set;
    description.resources[1].binding = description.resources[0].binding;
    auto workload = Workload::fromComputeShader(std::move(description));
    auto context = wrappedContext();

    Session session(context, workload);

    EXPECT_THROW(session.configure(), std::runtime_error);
}

TEST_F(StandaloneComputeSessionExecutionTest, RejectsMismatchedBindingResourceKinds) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersWithSampledImagesDescription());
    auto context = wrappedContext();
    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();

    EXPECT_THROW(bindings.bindTensor(workload.resource(0), TensorBindingInfo{nullptr}), std::runtime_error);
    EXPECT_THROW(bindings.bindBuffer(workload.resource(3), BufferBindingInfo{nullptr}), std::runtime_error);
    EXPECT_THROW(bindings.bindImage(workload.resource(0), {}), std::runtime_error);
}

TEST_F(StandaloneComputeSessionExecutionTest, PrepareRejectsMissingBindings) {
    constexpr std::size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    auto workload = Workload::fromComputeShader(makeAddInt32BuffersDescription());
    auto context = wrappedContext();

    const Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    const Buffer secondInputBuffer(physicalDevice, device, bufferSize);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();
    bindings.bindBuffer(workload.resource(0), BufferBindingInfo{*firstInputBuffer.buffer});
    bindings.bindBuffer(workload.resource(1), BufferBindingInfo{*secondInputBuffer.buffer});

    EXPECT_THROW((void)session.prepare(bindings), std::runtime_error);
}

TEST_F(StandaloneComputeSessionExecutionTest, RejectsImageSamplerPolicyViolations) {
    auto workload = Workload::fromComputeShader(makeAddInt32BuffersWithSampledImagesDescription());
    auto context = wrappedContext();

    auto runtimeSamplerImage = context.createImage(workload.resource(3));
    auto callerSamplerImage = context.createImage(workload.resource(4));
    auto sampler = makeNearestSampler(device);

    Session session(context, workload);
    session.configure();
    auto bindings = session.createBindingSet();

    auto runtimeSamplerBinding = runtimeSamplerImage.binding();
    runtimeSamplerBinding.sampler = *sampler;
    EXPECT_THROW(bindings.bindImage(workload.resource(3), runtimeSamplerBinding), std::runtime_error);

    auto callerSamplerBinding = callerSamplerImage.binding();
    EXPECT_THROW(bindings.bindImage(workload.resource(4), callerSamplerBinding), std::runtime_error);
}
