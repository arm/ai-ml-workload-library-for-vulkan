/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlworkloadlib/workload.hpp"

#include "internal/utils.hpp"
#include "internal/workload_builder.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace mlworkloadlib {

namespace utils = detail::utils;

using WorkloadBuilder = detail::WorkloadBuilder;

/*******************************************************************************
 * Standalone data graph workload
 *******************************************************************************/

Workload Workload::fromDataGraph(DataGraphDescription description) {
    WorkloadBuilder builder;

    // Module and executable
    const auto moduleIndex = builder.addModule(std::move(description.module), "data_graph", description.entryPoint);
    const auto executableIndex = builder.addExecutable("data_graph", ExecutableKind::Graph, moduleIndex);
    auto &executable = builder.executable(executableIndex);

    // Pipeline metadata
    utils::validateSpecializationInfo(description.pipeline.specializationInfo, "Standalone data graph pipeline");
    builder.setSpecializationInfo(executableIndex, std::move(description.pipeline.specializationInfo));
    executable.dataGraphPipelineIdentifier = std::move(description.pipeline.identifier);
    executable.dataGraphPipelineFlags = description.pipeline.flags;
    builder.setImplicitBarrier(executableIndex, description.implicitBarrier);

    // Public resources and descriptor bindings
    executable.bindings.reserve(description.resources.size());
    for (auto &resourceDescription : description.resources) {
        const auto resourceIndex =
            builder.addResource(std::move(resourceDescription.name), resourceDescription.resource,
                                WorkloadBuilder::publicRoleForAccess(resourceDescription.access));
        builder.addDescriptorBinding(executableIndex, resourceIndex, resourceDescription.set,
                                     resourceDescription.binding, resourceDescription.access);
    }

    // Graph constants
    executable.constantIndexes.reserve(description.constants.size());
    for (auto &constantDescription : description.constants) {
        const auto sparsityDimension =
            constantDescription.sparse2To4.has_value() ? constantDescription.sparse2To4->dimension : -1;
        if (constantDescription.sparse2To4.has_value() && !utils::isSparsityDimensionSpecified(sparsityDimension)) {
            throw std::runtime_error("Sparse standalone data graph constants must specify a sparsity dimension");
        }

        const auto resourceIndex =
            builder.addConstantResource(std::move(constantDescription.name), constantDescription.resource);
        const auto constantIndex =
            builder.addConstant(resourceIndex, constantDescription.data, constantDescription.size, sparsityDimension);
        executable.constantIndexes.push_back(constantIndex);
    }

    return builder.finish();
}

} // namespace mlworkloadlib
