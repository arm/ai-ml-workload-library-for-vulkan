/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"

#include "mlworkloadlib_utils/workload_metadata.hpp"

#include <exception>
#include <iostream>

namespace {

using namespace mlsdk::workloadlib;
using namespace mlsdk::workloadlib::samples;

// [resource-inspection-begin]
void printResource(ResourceView resource) {
    const auto requirements = resource.requirements();
    std::cout << "  [" << resource.index() << "] " << resource.name() << ": "
              << utils::resourceKindName(requirements.kind()) << ", " << utils::accessName(resource.access());

    switch (requirements.kind()) {
    case ResourceKind::Tensor: {
        std::cout << ", shape=[";
        const auto shape = requirements.asTensor().shape();
        for (std::size_t i = 0; i < shape.size(); ++i) {
            std::cout << (i == 0 ? "" : ", ") << shape[i];
        }
        std::cout << "]";
        break;
    }
    case ResourceKind::StorageBuffer:
        std::cout << ", bytes=" << requirements.byteSize();
        break;
    case ResourceKind::Image: {
        const auto extent = requirements.asImage().extent();
        std::cout << ", extent=" << extent.width << "x" << extent.height << "x" << extent.depth;
        break;
    }
    case ResourceKind::Unknown:
        break;
    }
    std::cout << '\n';
}
// [resource-inspection-end]

// [executable-inspection-begin]
void printExecutable(ExecutableView executable) {
    const auto module = executable.module();
    std::cout << "  [" << executable.index() << "] " << executable.name() << ": "
              << utils::executableKindName(executable.type()) << ", module=" << module.name()
              << ", entry-point=" << module.entryPoint() << '\n';

    for (uint32_t i = 0; i < executable.interfaceDescriptorBindingCount(); ++i) {
        const auto binding = executable.interfaceDescriptorBinding(i);
        std::cout << "      set=" << binding.set << ", binding=" << binding.binding
                  << ", resource=" << binding.resourceIndex << ", access=" << utils::accessName(binding.access) << '\n';
    }
}
// [executable-inspection-end]

} // namespace

int main() {
    try {
        // [in-memory-vgf-loading-begin]
        const auto vgf = addBuffersVgf(4);
        const auto workload = Workload::fromVGF(vgf.data(), vgf.size());
        // [in-memory-vgf-loading-end]

        std::cout << "Resources (" << workload.resourceCount() << "):\n";
        for (const auto resource : workload.resources()) {
            printResource(resource);
        }

        std::cout << "Executables (" << workload.executableCount() << "):\n";
        for (const auto executable : workload.executables()) {
            printExecutable(executable);
        }

        std::cout << "Placeholder modules: " << workload.placeholderModuleCount() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Failed to inspect workload: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
