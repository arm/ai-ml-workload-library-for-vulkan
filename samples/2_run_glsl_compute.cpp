/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib/session.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace mlworkloadlib;
    using namespace mlworkloadlib::samples;

    try {
        // [glsl-support-check-begin]
        if (!supports(Feature::GlslModules)) {
            throw std::runtime_error("The sample requires GLSL module support");
        }
        // [glsl-support-check-end]

        // [standalone-compute-execution-begin]
        const std::vector<int32_t> lhs = {1, 2, 3, 4};
        const std::vector<int32_t> rhs = {10, 20, 30, 40};

        const auto workload = Workload::fromComputeShader(addBuffersDescription(lhs.size()));
        auto context = Context::create();
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
        execution.run();
        // [standalone-compute-execution-end]

        const std::vector<int32_t> expected = {11, 22, 33, 44};
        const auto output = readMemory<int32_t>(contextView.device, outputBuffer.memory(), lhs.size());
        if (output != expected) {
            std::cerr << "Unexpected output\n";
            return 1;
        }

        std::cout << "Output:";
        for (const auto value : output) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Failed to run compute sample: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
