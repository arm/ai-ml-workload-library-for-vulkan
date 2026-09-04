/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"
#include "sample_vulkan.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    using namespace mlworkloadlib;
    using namespace mlworkloadlib::samples;

    try {
        const std::vector<int32_t> lhs = {1, 2, 3, 4};
        const std::vector<int32_t> rhs = {10, 20, 30, 40};

        const auto workload = Workload::fromComputeShader(addBuffersDescription(lhs.size()));

        // [wrapped-context-begin]
        const ApplicationVulkanContext applicationVulkan;
        auto context = Context::wrap(applicationVulkan.view());
        // [wrapped-context-end]
        const auto contextView = context.contextView();

        auto lhsBuffer = context.createBuffer(workload.resource(0));
        auto rhsBuffer = context.createBuffer(workload.resource(1));
        auto outputBuffer = context.createBuffer(workload.resource(2));

        writeMemory(contextView.device, lhsBuffer.memory(), lhs);
        writeMemory(contextView.device, rhsBuffer.memory(), rhs);
        clearMemory(contextView.device, outputBuffer.memory());

        Session session(context, workload);
        session.configure();
        auto bindings = session.createBindingSet();
        bindings.bindBuffer(workload.resource(0), {lhsBuffer.handle(), lhsBuffer.memory()});
        bindings.bindBuffer(workload.resource(1), {rhsBuffer.handle(), rhsBuffer.memory()});
        bindings.bindBuffer(workload.resource(2), {outputBuffer.handle(), outputBuffer.memory()});
        auto execution = session.prepare(bindings);

        // [recorded-execution-begin]
        const auto &device = contextView.device.get();
        const auto &queue = contextView.queue.get();
        const vk::raii::CommandPool commandPool(
            device, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, contextView.queueFamilyIndex});
        auto commandBuffer =
            std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
        const vk::raii::Fence fence(device, vk::FenceCreateInfo{});

        commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        execution.record(*commandBuffer);
        commandBuffer.end();

        const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
        queue.submit(submitInfo, *fence);
        if (device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("Timed out waiting for the recorded workload");
        }
        // [recorded-execution-end]

        const std::vector<int32_t> expected = {11, 22, 33, 44};
        const auto output = readMemory<int32_t>(contextView.device, outputBuffer.memory(), lhs.size());
        if (output != expected) {
            std::cerr << "Unexpected output\n";
            return 1;
        }

        std::cout << "Wrapped-context output:";
        for (const auto value : output) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Failed to run wrapped-context sample: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
