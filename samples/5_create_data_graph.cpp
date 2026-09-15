/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlworkloadlib/workload.hpp"
#include "mlworkloadlib_utils/workload_metadata.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <utility>

int main() {
    using namespace mlsdk::workloadlib;

    try {
        // [data-graph-description-begin]
        DataGraphDescription description;
        description.module.codeKind = ModuleCodeKind::Missing;
        description.entryPoint = "main";
        description.resources = {
            {"input", 0, 0, ResourceAccess::Read, utils::tensorRequirements(vk::Format::eR8Sint, {1, 16, 16, 16})},
            {"output", 1, 1, ResourceAccess::Write, utils::tensorRequirements(vk::Format::eR8Sint, {1, 8, 8, 16})},
        };
        description.pipeline.identifier = "maxpool";

        const auto workload = Workload::fromDataGraph(std::move(description));
        // [data-graph-description-end]

        std::cout << "Created a standalone data-graph workload with " << workload.resourceCount()
                  << " public resources and " << workload.placeholderModuleCount() << " placeholder module\n";
    } catch (const std::exception &error) {
        std::cerr << "Failed to create data-graph workload: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
