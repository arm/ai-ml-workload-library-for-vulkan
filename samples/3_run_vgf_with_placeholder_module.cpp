/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"
#include "mlworkloadlib_utils/mapped_device_memory.hpp"

#include <vgf/encoder.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::samples;

// [vgf-building-begin]
std::string addBuffersVgf(std::size_t elementCount) {
    if (elementCount > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("The element count exceeds the VGF dispatch limit");
    }

    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    const auto module = encoder->AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main");

    const auto shape = std::vector<int64_t>{static_cast<int64_t>(elementCount)};
    const auto strides = std::vector<int64_t>{static_cast<int64_t>(sizeof(int32_t))};
    const auto lhs = encoder->AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, shape, strides);
    const auto rhs = encoder->AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, shape, strides);
    const auto output =
        encoder->AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, shape, strides);

    const auto lhsBinding = encoder->AddBindingSlot(0, lhs);
    const auto rhsBinding = encoder->AddBindingSlot(1, rhs);
    const auto outputBinding = encoder->AddBindingSlot(2, output);
    const auto descriptorSet = encoder->AddDescriptorSetInfo({lhsBinding, rhsBinding, outputBinding}, 0);
    const std::vector<mlsdk::vgflib::GraphConstantBindingRef> noGraphConstants;
    encoder->AddSegmentInfo(module, "add_int32_buffers_segment", {descriptorSet}, {lhsBinding, rhsBinding},
                            {outputBinding}, noGraphConstants, {static_cast<uint32_t>(elementCount), 1, 1});
    encoder->AddModelSequenceInputsOutputs({lhsBinding, rhsBinding}, {"lhs", "rhs"}, {outputBinding}, {"output"});
    encoder->Finish();

    std::stringstream stream;
    if (!encoder->WriteTo(stream)) {
        throw std::runtime_error("Failed to encode the sample VGF");
    }
    return stream.str();
}
// [vgf-building-end]

// [runtime-resource-binding-begin]
struct RuntimeResources {
    std::vector<TensorAllocation> tensors;
    std::vector<BufferAllocation> buffers;
    std::vector<ImageAllocation> images;
};

RuntimeResources bindRuntimeResources(Context &context, const Workload &workload, BindingSet &bindings) {
    RuntimeResources allocations;
    const auto contextView = context.contextView();

    for (const auto resource : workload.resources()) {
        const auto requirements = resource.requirements();
        if (requirements.participatesInAliasing()) {
            throw std::runtime_error("The sample cannot allocate aliased resource " + std::to_string(resource.index()));
        }

        switch (requirements.kind()) {
        case ResourceKind::Tensor: {
            auto allocation = context.createTensor(resource);
            utils::clearDeviceMemory(contextView.device, allocation.memory());
            bindings.bindTensor(resource, {allocation.handle(), allocation.memory()});
            allocations.tensors.push_back(std::move(allocation));
            break;
        }
        case ResourceKind::StorageBuffer: {
            auto allocation = context.createBuffer(resource);
            utils::clearDeviceMemory(contextView.device, allocation.memory());
            bindings.bindBuffer(resource, {allocation.handle(), allocation.memory()});
            allocations.buffers.push_back(std::move(allocation));
            break;
        }
        case ResourceKind::Image: {
            if (requirements.asImage().requiresSamplerBinding()) {
                throw std::runtime_error("The sample cannot provide a caller-owned sampler for image resource " +
                                         std::to_string(resource.index()));
            }
            auto allocation = context.createImage(resource);
            bindings.bindImage(resource, allocation.binding());
            allocations.images.push_back(std::move(allocation));
            break;
        }
        case ResourceKind::Unknown:
            throw std::runtime_error("Unsupported resource kind at index " + std::to_string(resource.index()));
        }
    }
    return allocations;
}
// [runtime-resource-binding-end]

} // namespace

int main() {
    try {
        const std::vector<int32_t> lhs = {1, 2, 3, 4};
        const std::vector<int32_t> rhs = {10, 20, 30, 40};

        // [in-memory-vgf-loading-begin]
        const auto vgf = addBuffersVgf(lhs.size());
        const auto workload = Workload::fromVGF(vgf.data(), vgf.size());
        // [in-memory-vgf-loading-end]
        if (workload.placeholderModuleCount() != 1) {
            throw std::runtime_error("The sample expects one placeholder VGF module");
        }

        auto context = Context::create();
        Session session(context, workload);

        // [placeholder-module-binding-begin]
        ModuleImplementation implementation;
        implementation.codeKind = ModuleCodeKind::Glsl;
        implementation.source = addBuffersGlsl;
        session.bindModule(workload.placeholderModule(0), std::move(implementation));
        session.configure();
        // [placeholder-module-binding-end]

        auto bindings = session.createBindingSet();
        const auto allocations = bindRuntimeResources(context, workload, bindings);
        if (allocations.buffers.size() != 3) {
            throw std::runtime_error("The sample expects three VGF buffer resources");
        }

        const auto contextView = context.contextView();
        utils::writeDeviceMemory(contextView.device, allocations.buffers.at(0).memory(), lhs);
        utils::writeDeviceMemory(contextView.device, allocations.buffers.at(1).memory(), rhs);
        utils::clearDeviceMemory(contextView.device, allocations.buffers.at(2).memory());

        auto execution = session.prepare(bindings);
        execution.run();

        const std::vector<int32_t> expected = {11, 22, 33, 44};
        const auto output =
            utils::readDeviceMemory<int32_t>(contextView.device, allocations.buffers.at(2).memory(), lhs.size());
        if (output != expected) {
            std::cerr << "Unexpected output\n";
            return 1;
        }

        std::cout << "VGF output:";
        for (const auto value : output) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Failed to run VGF workload: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
