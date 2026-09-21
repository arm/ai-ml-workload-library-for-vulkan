/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/hlsl_compiler.hpp"
#include "internal/workload_impl.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace mlsdk::workloadlib::detail {

TEST(HlslCompiler, CompilesRepeatedlyAfterErrors) {
    Module module;
    module.name = "allocator-test.hlsl";
    module.entryPoint = "main";
    module.codeKind = ModuleCodeKind::Hlsl;
    for (int iteration = 0; iteration < 3; ++iteration) {
        module.source = "invalid HLSL";
        EXPECT_THROW(compileHlslComputeToSpirv(module), std::runtime_error);

        module.source = R"(
            RWStructuredBuffer<uint> output : register(u0);
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID) { output[id.x] = id.x + 1; }
        )";
        const auto spirv = compileHlslComputeToSpirv(module);
        ASSERT_FALSE(spirv.empty());
        EXPECT_EQ(spirv.front(), 0x07230203u);
    }
}

} // namespace mlsdk::workloadlib::detail
