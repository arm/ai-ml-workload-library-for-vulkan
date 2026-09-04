/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "workload_fuzzer.hpp"

#include "mlworkloadlib/workload.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>

namespace mlworkloadlib::fuzz {
namespace {

constexpr std::size_t kMaxRawInputSize = 64U * 1024U;

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "mlworkloadlib fuzzer invariant failed: %s\n", message);
    std::abort();
}

[[noreturn]] void failException(const char *context, const char *message) {
    std::fprintf(stderr, "mlworkloadlib fuzzer invariant failed: %s: %s\n", context, message);
    std::abort();
}

void expect(bool condition, const char *message) {
    if (!condition) {
        fail(message);
    }
}

ResourceKind kindForDescriptor(vk::DescriptorType descriptorType) {
    switch (descriptorType) {
    case vk::DescriptorType::eTensorARM:
        return ResourceKind::Tensor;
    case vk::DescriptorType::eStorageBuffer:
        return ResourceKind::StorageBuffer;
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage:
        return ResourceKind::Image;
    default:
        return ResourceKind::Unknown;
    }
}

bool isValidAccess(ResourceAccess access) {
    switch (access) {
    case ResourceAccess::Read:
    case ResourceAccess::Write:
    case ResourceAccess::ReadWrite:
        return true;
    }
    return false;
}

bool isValidExecutableKind(ExecutableKind type) {
    switch (type) {
    case ExecutableKind::Graph:
    case ExecutableKind::Compute:
        return true;
    }
    return false;
}

void expectResourceOutOfRange(const Workload &workload) {
    bool threw = false;
    try {
        static_cast<void>(workload.resource(workload.resourceCount()));
    } catch (const std::out_of_range &) {
        threw = true;
    } catch (const std::exception &error) {
        failException("unexpected resource out-of-range exception", error.what());
    }
    expect(threw, "resource(resourceCount()) must throw std::out_of_range");
}

void expectExecutableOutOfRange(const Workload &workload) {
    bool threw = false;
    try {
        static_cast<void>(workload.executable(workload.executableCount()));
    } catch (const std::out_of_range &) {
        threw = true;
    } catch (const std::exception &error) {
        failException("unexpected executable out-of-range exception", error.what());
    }
    expect(threw, "executable(executableCount()) must throw std::out_of_range");
}

void expectBindingOutOfRange(ExecutableView executable, uint32_t bindingCount) {
    bool threw = false;
    try {
        static_cast<void>(executable.interfaceDescriptorBinding(bindingCount));
    } catch (const std::out_of_range &) {
        threw = true;
    } catch (const std::exception &error) {
        failException("unexpected binding out-of-range exception", error.what());
    }
    expect(threw, "interfaceDescriptorBinding(count) must throw std::out_of_range");
}

void verifyResources(const Workload &workload) {
    const auto resourceCount = workload.resourceCount();
    uint32_t rangedResourceCount = 0;
    for (const auto resource : workload.resources()) {
        static_cast<void>(resource);
        ++rangedResourceCount;
    }
    expect(rangedResourceCount == resourceCount, "resource range count must match resourceCount()");

    for (uint32_t resourceIndex = 0; resourceIndex < resourceCount; ++resourceIndex) {
        const auto resource = workload.resource(resourceIndex);
        static_cast<void>(resource.name());
        const auto access = resource.access();
        expect(isValidAccess(access), "resource access must be a known enum value");
        expect(resource.access() == access, "resource access must be deterministic");

        const auto requirements = resource.requirements();
        const auto kind = requirements.kind();
        const auto descriptorType = requirements.descriptorType();
        expect(kind == kindForDescriptor(descriptorType), "resource kind must match descriptor type");
        static_cast<void>(requirements.format());
        if (requirements.participatesInAliasing()) {
            expect(requirements.requiresBoundMemoryInfo(), "aliased resources must require bound memory info");
        }

        const auto elementCount = requirements.elementCount();
        const auto byteSize = requirements.byteSize();
        expect(requirements.elementCount() == elementCount, "elementCount() must be deterministic");
        expect(requirements.byteSize() == byteSize, "byteSize() must be deterministic");
    }

    expectResourceOutOfRange(workload);
}

void verifyExecutables(const Workload &workload) {
    const auto executableCount = workload.executableCount();
    uint32_t rangedExecutableCount = 0;
    for (const auto executable : workload.executables()) {
        expect(executable.index() == rangedExecutableCount, "executable range index must be sequential");
        ++rangedExecutableCount;
    }
    expect(rangedExecutableCount == executableCount, "executable range count must match executableCount()");

    for (uint32_t executableIndex = 0; executableIndex < executableCount; ++executableIndex) {
        const auto executable = workload.executable(executableIndex);
        expect(executable.index() == executableIndex, "executable(index).index() must match index");
        expect(isValidExecutableKind(executable.type()), "executable type must be a known enum value");
        static_cast<void>(executable.name());
        const auto module = executable.module();
        static_cast<void>(module.index());
        static_cast<void>(module.name());
        static_cast<void>(module.entryPoint());

        const auto bindingCount = executable.interfaceDescriptorBindingCount();
        for (uint32_t bindingIndex = 0; bindingIndex < bindingCount; ++bindingIndex) {
            const auto binding = executable.interfaceDescriptorBinding(bindingIndex);
            expect(binding.resourceIndex < workload.resourceCount(),
                   "interface binding must reference a public resource");
            expect(isValidAccess(binding.access), "interface binding access must be a known enum value");
            expect(binding.kind == kindForDescriptor(binding.descriptorType),
                   "interface binding kind must match descriptor type");

            const auto resource = workload.resource(binding.resourceIndex);
            const auto requirements = resource.requirements();
            expect(binding.kind == requirements.kind(), "interface binding kind must match referenced resource");
            expect(binding.descriptorType == requirements.descriptorType(),
                   "interface binding descriptor type must match referenced resource");
        }
        expectBindingOutOfRange(executable, bindingCount);
    }

    expectExecutableOutOfRange(workload);
}

void verifyWorkload(const Workload &workload) {
    verifyResources(workload);
    verifyExecutables(workload);
}

} // namespace

void fuzzWorkloadBytes(const uint8_t *data, std::size_t size) {
    if (data == nullptr || size == 0 || size > kMaxRawInputSize) {
        return;
    }

    bool imported = false;
    try {
        auto workload = Workload::fromVGF(data, size);
        imported = true;
        verifyWorkload(workload);
    } catch (const std::out_of_range &error) {
        if (imported) {
            failException("post-import std::out_of_range", error.what());
        }
    } catch (const std::logic_error &error) {
        if (imported) {
            failException("post-import std::logic_error", error.what());
        }
    } catch (const std::runtime_error &error) {
        if (imported) {
            failException("post-import std::runtime_error", error.what());
        }
    } catch (const std::exception &error) {
        failException("unexpected std::exception", error.what());
    } catch (...) {
        fail("unexpected non-standard exception");
    }
}

} // namespace mlworkloadlib::fuzz
