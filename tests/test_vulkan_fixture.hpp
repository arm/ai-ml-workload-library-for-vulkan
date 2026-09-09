/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/context.hpp"
#include "mlworkloadlib_utils/application_context.hpp"

#include <gtest/gtest.h>

namespace mlsdk::workloadlib::test {

/*******************************************************************************
 * Vulkan fixture
 *******************************************************************************/

class RuntimeSessionExecutionTest : public ::testing::Test, protected utils::ApplicationContext {
  protected:
    RuntimeSessionExecutionTest() : utils::ApplicationContext(utils::ApplicationContext::DeferredInitialization{}) {}

    void SetUp() override {
        if (!initialize("mlworkloadlib-test")) {
            GTEST_SKIP() << "No Vulkan device with required data graph extensions and compute queue support";
        }
    }

    Context wrappedContext() { return Context::wrap({instance, physicalDevice, device, queueFamilyIndex, queue}); }
};

} // namespace mlsdk::workloadlib::test
