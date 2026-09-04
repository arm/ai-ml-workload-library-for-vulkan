/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlworkloadlib/workload.hpp"

#include "internal/utils.hpp"
#include "internal/workload_builder.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace mlworkloadlib {

namespace utils = detail::utils;

using WorkloadBuilder = detail::WorkloadBuilder;

/*******************************************************************************
 * Standalone compute workload
 *******************************************************************************/

Workload Workload::fromComputeShader(ComputeShaderDescription description) {
    WorkloadBuilder builder;

    // Module and executable
    const auto moduleIndex = builder.addModule(std::move(description.module), "compute", description.entryPoint);
    const auto executableIndex = builder.addExecutable("compute", ExecutableKind::Compute, moduleIndex);
    auto &executable = builder.executable(executableIndex);

    // Execution controls
    builder.setDispatchShape(executableIndex, description.dispatch);
    if (description.pushConstantSize != 0) {
        builder.addPushConstantRange(executableIndex, vk::ShaderStageFlagBits::eCompute, 0,
                                     description.pushConstantSize);
    }
    builder.setImplicitBarrier(executableIndex, description.implicitBarrier);

    // Specialization constants
    utils::validateSpecializationInfo(description.specializationInfo, "Standalone compute");
    builder.setSpecializationInfo(executableIndex, std::move(description.specializationInfo));

    // Public resources and descriptor bindings
    executable.bindings.reserve(description.resources.size());
    for (auto &resourceDescription : description.resources) {
        const auto resourceIndex =
            builder.addResource(std::move(resourceDescription.name), resourceDescription.resource,
                                WorkloadBuilder::publicRoleForAccess(resourceDescription.access));
        builder.addDescriptorBinding(executableIndex, resourceIndex, resourceDescription.set,
                                     resourceDescription.binding, resourceDescription.access);
    }

    return builder.finish();
}

} // namespace mlworkloadlib
