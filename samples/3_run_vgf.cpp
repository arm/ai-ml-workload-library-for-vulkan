/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace mlworkloadlib;
using namespace mlworkloadlib::samples;

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
            clearMemory(contextView.device, allocation.memory());
            bindings.bindTensor(resource, {allocation.handle(), allocation.memory()});
            allocations.tensors.push_back(std::move(allocation));
            break;
        }
        case ResourceKind::StorageBuffer: {
            auto allocation = context.createBuffer(resource);
            clearMemory(contextView.device, allocation.memory());
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
        const auto vgf = addBuffersVgf(4, false);
        const auto workload = Workload::fromVGF(vgf.data(), vgf.size());
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
        [[maybe_unused]] const auto allocations = bindRuntimeResources(context, workload, bindings);
        auto execution = session.prepare(bindings);
        execution.run();

        std::cout << "Executed " << workload.executableCount() << " VGF executable(s) with " << workload.resourceCount()
                  << " public resource(s)\n";
    } catch (const std::exception &error) {
        std::cerr << "Failed to run VGF workload: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
