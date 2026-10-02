/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/binding_set_impl.hpp"
#include "internal/module_compiler.hpp"
#include "internal/session_impl.hpp"
#include "internal/utils.hpp"
#include "internal/workload_builder.hpp"

#include "mlworkloadlib/session.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlsdk::workloadlib {

using DescriptorBinding = detail::DescriptorBinding;
using Executable = detail::Executable;
using Module = detail::Module;

namespace {

/*******************************************************************************
 * Pipeline creation helpers
 *******************************************************************************/

struct SpecializationInfoStorage {
    vk::SpecializationInfo info;

    // vk::SpecializationInfo borrows executable specialization storage until pipeline creation returns.
    const vk::SpecializationInfo *populate(const SpecializationInfo &specializationInfo) {
        if (specializationInfo.empty()) {
            return nullptr;
        }

        info.mapEntryCount = static_cast<uint32_t>(specializationInfo.mapEntries.size());
        info.pMapEntries = specializationInfo.mapEntries.data();
        info.dataSize = specializationInfo.data.size();
        info.pData = specializationInfo.data.data();
        return &info;
    }
};

void validateUniqueDescriptorSlots(const Executable &executable) {
    auto sortedBindings = executable.bindings;
    std::sort(sortedBindings.begin(), sortedBindings.end(),
              [](const DescriptorBinding &lhs, const DescriptorBinding &rhs) {
                  return lhs.set != rhs.set ? lhs.set < rhs.set : lhs.binding < rhs.binding;
              });

    const auto duplicateBinding = std::adjacent_find(sortedBindings.begin(), sortedBindings.end(),
                                                     [](const DescriptorBinding &lhs, const DescriptorBinding &rhs) {
                                                         return lhs.set == rhs.set && lhs.binding == rhs.binding;
                                                     });
    if (duplicateBinding != sortedBindings.end()) {
        throw std::runtime_error("Workload executable '" + executable.name + "' has duplicate descriptor set " +
                                 std::to_string(duplicateBinding->set) + " binding " +
                                 std::to_string(duplicateBinding->binding));
    }
}

void validateSupportedDescriptorType(vk::DescriptorType descriptorType) {
    switch (descriptorType) {
    case vk::DescriptorType::eStorageBuffer:
    case vk::DescriptorType::eTensorARM:
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eStorageImage:
        return;
    default:
        throw std::runtime_error("Session does not support descriptor type " +
                                 std::to_string(static_cast<uint32_t>(descriptorType)));
    }
}

struct DataGraphPipelineCreateStorage {
    std::vector<vk::TensorDescriptionARM> resourceTensorDescriptions;
    std::vector<vk::DataGraphPipelineResourceInfoImageLayoutARM> imageLayouts;
    std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
    std::vector<vk::TensorDescriptionARM> constantTensorDescriptions;
    std::vector<vk::DataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM> sparsityInfos;
    std::vector<vk::DataGraphPipelineConstantARM> constants;
};

void validateExecutableBindings(const Workload &workload, const Executable &executable) {
    validateUniqueDescriptorSlots(executable);

    const auto &workloadState = workloadImpl(workload);
    for (const auto &descBinding : executable.bindings) {
        const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
        validateSupportedDescriptorType(descriptorType);
        if (descriptorType == vk::DescriptorType::eCombinedImageSampler ||
            descriptorType == vk::DescriptorType::eStorageImage) {
            const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
            if (resource.format != vk::Format::eR8G8B8A8Snorm) {
                throw std::runtime_error("Session only supports eR8G8B8A8Snorm image resources");
            }
        }
    }
}

void populateDataGraphResources(DataGraphPipelineCreateStorage &storage, const Executable &executable,
                                const Workload &workload) {
    const auto &workloadState = workloadImpl(workload);
    storage.resourceTensorDescriptions.reserve(executable.bindings.size());
    storage.imageLayouts.reserve(executable.bindings.size());
    storage.resourceInfos.reserve(executable.bindings.size());
    for (const auto &descBinding : executable.bindings) {
        const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
        const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
        const auto tensorTiling = descriptorType == vk::DescriptorType::eCombinedImageSampler ||
                                          descriptorType == vk::DescriptorType::eStorageImage
                                      ? vk::TensorTilingARM::eOptimal
                                      : vk::TensorTilingARM::eLinear;
        storage.resourceTensorDescriptions.emplace_back(
            tensorTiling, resource.format, static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
            resource.stride.empty() ? nullptr : resource.stride.data(), vk::TensorUsageFlagBitsARM::eDataGraph);
        if (descriptorType == vk::DescriptorType::eCombinedImageSampler ||
            descriptorType == vk::DescriptorType::eStorageImage) {
            storage.imageLayouts.emplace_back(detail::imageLayout(descriptorType),
                                              &storage.resourceTensorDescriptions.back());
            storage.resourceInfos.emplace_back(descBinding.set, descBinding.binding, 0, &storage.imageLayouts.back());
        } else {
            storage.resourceInfos.emplace_back(descBinding.set, descBinding.binding, 0,
                                               &storage.resourceTensorDescriptions.back());
        }
    }
}

void populateDataGraphConstants(DataGraphPipelineCreateStorage &storage, const Executable &executable,
                                const Workload &workload) {
    const auto &workloadState = workloadImpl(workload);
    const auto numConstants = static_cast<uint32_t>(executable.constantIndexes.size());
    storage.constantTensorDescriptions.reserve(numConstants);
    storage.sparsityInfos.reserve(numConstants);
    storage.constants.reserve(numConstants);
    for (uint32_t constantIndex = 0; constantIndex < numConstants; ++constantIndex) {
        const auto workloadConstantIndex = executable.constantIndexes.at(constantIndex);
        const auto &constant = workloadState.constants.at(workloadConstantIndex);
        const auto &resource = workloadState.resources.at(constant.resourceIndex);
        void *pNext = nullptr;
        if (constant.sparsityDimension < detail::Constant::unspecifiedSparsityDimension) {
            throw std::runtime_error("Graph constant has invalid sparsity dimension");
        }
        if (constant.hasSparsity()) {
            constexpr uint32_t zeroCount = 2;
            constexpr uint32_t groupSize = 4;
            storage.sparsityInfos.emplace_back(static_cast<uint32_t>(constant.sparsityDimension), zeroCount, groupSize,
                                               nullptr);
            pNext = &storage.sparsityInfos.back();
        }
        storage.constantTensorDescriptions.emplace_back(
            vk::TensorTilingARM::eLinear, resource.format, static_cast<uint32_t>(resource.shape.size()),
            resource.shape.data(), resource.stride.empty() ? nullptr : resource.stride.data(),
            vk::TensorUsageFlagBitsARM::eDataGraph, pNext);
        storage.constants.emplace_back(workloadConstantIndex, constant.payloadView.data(),
                                       &storage.constantTensorDescriptions.back());
    }
}

/*******************************************************************************
 * Module resolution
 *******************************************************************************/

Module moduleForExecutable(const Workload &workload, const std::map<uint32_t, Module> &moduleImplementations,
                           uint32_t executableIndex) {
    const auto &workloadState = workloadImpl(workload);
    const auto &executable = workloadState.executables.at(executableIndex);
    const auto &workloadModule = workloadState.modules.at(executable.moduleIndex);
    if (workloadModule.codeKind != ModuleCodeKind::Missing) {
        return workloadModule;
    }

    const auto moduleImplementationIt = moduleImplementations.find(executable.moduleIndex);
    if (moduleImplementationIt == moduleImplementations.end()) {
        throw std::runtime_error("Workload module '" + workloadModule.name +
                                 "' requires an application-supplied implementation");
    }
    return moduleImplementationIt->second;
}

/*******************************************************************************
 * Executable configuration
 *******************************************************************************/

std::vector<vk::raii::DescriptorSetLayout>
createDescriptorSetLayouts(const std::vector<DescriptorBinding> &descBindings, const Workload &workload,
                           const ContextView &contextView) {
    const auto &workloadState = workloadImpl(workload);
    const auto descBindingSets = [&descBindings] {
        std::vector<std::vector<DescriptorBinding>> sets;
        for (const auto &descBinding : descBindings) {
            while (sets.size() <= descBinding.set) {
                sets.emplace_back();
            }
            sets[descBinding.set].push_back(descBinding);
        }
        return sets;
    }();

    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
    descriptorSetLayouts.reserve(descBindingSets.size());
    for (const auto &setDescBindings : descBindingSets) {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(setDescBindings.size());
        for (const auto &descBinding : setDescBindings) {
            layoutBindings.emplace_back(descBinding.binding, workloadState.descriptorTypeForBinding(descBinding), 1,
                                        vk::ShaderStageFlagBits::eAll);
        }
        descriptorSetLayouts.emplace_back(contextView.device.get(),
                                          vk::DescriptorSetLayoutCreateInfo({}, layoutBindings));
    }
    return descriptorSetLayouts;
}

void validatePushConstantRanges(const Executable &executable) {
    if (executable.pushConstantSize != 0 && executable.pushConstantRanges.empty()) {
        throw std::runtime_error("Workload declares push constants without push constant ranges");
    }
    for (const auto &range : executable.pushConstantRanges) {
        if (range.size == 0 || range.offset > executable.pushConstantSize ||
            range.size > executable.pushConstantSize - range.offset) {
            throw std::runtime_error("Workload push constant range is invalid");
        }
    }
}

vk::raii::PipelineLayout createPipelineLayout(const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts,
                                              const Executable &executable, const ContextView &contextView) {
    const auto rawDescriptorSetLayouts = detail::rawDescriptorSetLayouts(descriptorSetLayouts);
    validatePushConstantRanges(executable);
    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, rawDescriptorSetLayouts,
                                                                executable.pushConstantRanges);
    return {contextView.device.get(), pipelineLayoutCreateInfo};
}

vk::raii::ShaderModule createShaderModule(const Module &compiledModule, const ContextView &contextView) {
    const vk::ShaderModuleCreateInfo shaderCreateInfo({}, compiledModule.code.size() * sizeof(uint32_t),
                                                      compiledModule.code.data());
    return {contextView.device.get(), shaderCreateInfo};
}

vk::raii::Pipeline createComputePipeline(const Executable &executable, const Module &compiledModule,
                                         const vk::raii::ShaderModule &shaderModule,
                                         const vk::raii::PipelineLayout &pipelineLayout, const ContextView &contextView,
                                         const vk::raii::PipelineCache *pipelineCache) {
    const auto &dispatchShape = executable.dispatchShape;
    if (dispatchShape[0] == 0 || dispatchShape[1] == 0 || dispatchShape[2] == 0) {
        throw std::runtime_error("Compute shader executables must have a non-zero 3D dispatch shape");
    }

    SpecializationInfoStorage specializationInfo;
    const vk::PipelineShaderStageCreateInfo shaderStageCreateInfo(
        {}, vk::ShaderStageFlagBits::eCompute, *shaderModule, compiledModule.entryPoint.c_str(),
        specializationInfo.populate(executable.specializationInfo));
    const vk::ComputePipelineCreateInfo pipelineCreateInfo({}, shaderStageCreateInfo, *pipelineLayout);
    return {contextView.device.get(), pipelineCache, pipelineCreateInfo};
}

vk::raii::Pipeline createDataGraphPipeline(const Executable &executable, const Module &compiledModule,
                                           const vk::raii::ShaderModule &shaderModule,
                                           const vk::raii::PipelineLayout &pipelineLayout, const Workload &workload,
                                           const ContextView &contextView,
                                           const vk::raii::PipelineCache *pipelineCache) {
    DataGraphPipelineCreateStorage storage;
    populateDataGraphResources(storage, executable, workload);
    populateDataGraphConstants(storage, executable, workload);

    SpecializationInfoStorage specializationInfo;
    const vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleInfo(
        *shaderModule, compiledModule.entryPoint.c_str(), specializationInfo.populate(executable.specializationInfo),
        static_cast<uint32_t>(storage.constants.size()), storage.constants.data(), nullptr);
    const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo(executable.dataGraphPipelineFlags, *pipelineLayout,
                                                                static_cast<uint32_t>(storage.resourceInfos.size()),
                                                                storage.resourceInfos.data(), &shaderModuleInfo);
    const vk::raii::DeferredOperationKHR deferredOperation(nullptr);
    return {contextView.device.get(), deferredOperation, pipelineCache, pipelineCreateInfo};
}

/*******************************************************************************
 * Data graph session memory
 *******************************************************************************/

std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM>
allocateDataGraphSessionMemory(const ContextView &contextView,
                               const vk::raii::DataGraphPipelineSessionARM &graphSession,
                               std::vector<vk::raii::DeviceMemory> &sessionMemory) {
    const vk::DataGraphPipelineSessionBindPointRequirementsInfoARM bindPointInfo(*graphSession);
    const auto bindPointRequirements =
        contextView.device.get().getDataGraphPipelineSessionBindPointRequirementsARM(bindPointInfo);
    std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
    for (const auto &bindPointRequirement : bindPointRequirements) {
        if (bindPointRequirement.bindPointType != vk::DataGraphPipelineSessionBindPointTypeARM::eMemory) {
            continue;
        }

        for (uint32_t objectIndex = 0; objectIndex < bindPointRequirement.numObjects; ++objectIndex) {
            const vk::DataGraphPipelineSessionMemoryRequirementsInfoARM memoryInfo(
                *graphSession, bindPointRequirement.bindPoint, objectIndex);
            const auto memReqs = contextView.device.get().getDataGraphPipelineSessionMemoryRequirementsARM(memoryInfo);
            if (memReqs.memoryRequirements.size == 0) {
                continue;
            }

            const auto memoryType =
                detail::findMemoryType(contextView.physicalDevice.get(), memReqs.memoryRequirements.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal);
            const vk::MemoryAllocateInfo allocateInfo(memReqs.memoryRequirements.size, memoryType);
            sessionMemory.emplace_back(contextView.device.get(), allocateInfo);
            bindInfos.emplace_back(*graphSession, bindPointRequirement.bindPoint, objectIndex, *sessionMemory.back());
        }
    }
    return bindInfos;
}

} // namespace

/*******************************************************************************
 * Session implementation lifetime
 *******************************************************************************/

Session::Impl::Impl(Context &contextIn, const Workload &workloadIn, SessionOptions optionsIn)
    : workload(workloadIn), contextView(contextIn.contextImpl().contextView()),
      compiledExecutionState(std::make_shared<CompiledExecution::Impl>(contextIn, workloadIn, optionsIn)) {}

Session::Impl::Impl(Context &contextIn, const Workload &workloadIn, CompiledExecution &compiledExecutionIn)
    : workload(workloadIn), contextView(contextIn.contextImpl().contextView()),
      compiledExecutionState(compiledExecutionIn.compiledExecutionImpl()) {
    if (compiledExecutionState->context != &contextIn) {
        throw std::runtime_error("CompiledExecution was not created for this Context");
    }
    if (compiledExecutionState->workload != &workloadIn) {
        throw std::runtime_error("CompiledExecution was not created for this Workload");
    }
}

/*******************************************************************************
 * Compiled workload construction
 *******************************************************************************/

void Session::Impl::createPipeline(detail::CompiledExecutable &compiledExecutable, uint32_t executableIndex) const {
    const auto &executable = workloadImpl(workload).executables.at(executableIndex);
    if (executable.type != ExecutableKind::Graph && executable.type != ExecutableKind::Compute) {
        throw std::runtime_error("Session only supports data graph and compute shader executables");
    }

    // Module resolution
    auto compiledModule = moduleForExecutable(workload, moduleImplementations, executableIndex);
    detail::compileModuleToSpirv(compiledModule, executable.type);

    // Shared pipeline state
    validateExecutableBindings(workload, executable);
    compiledExecutable.descriptorSetLayouts = createDescriptorSetLayouts(executable.bindings, workload, contextView);
    compiledExecutable.pipelineLayout =
        createPipelineLayout(compiledExecutable.descriptorSetLayouts, executable, contextView);
    compiledExecutable.shaderModule = createShaderModule(compiledModule, contextView);

    // Executable-specific pipeline state
    if (executable.type == ExecutableKind::Compute) {
        compiledExecutable.pipeline = createComputePipeline(executable, compiledModule, compiledExecutable.shaderModule,
                                                            compiledExecutable.pipelineLayout, contextView,
                                                            compiledExecutionState->options.pipelineCache);
        return;
    }

    compiledExecutable.pipeline = createDataGraphPipeline(executable, compiledModule, compiledExecutable.shaderModule,
                                                          compiledExecutable.pipelineLayout, workload, contextView,
                                                          compiledExecutionState->options.pipelineCache);
}

/*******************************************************************************
 * Configuration
 *******************************************************************************/

void Session::Impl::configureExecutableState(uint32_t executableIndex) {
    const auto &executable = workloadImpl(workload).executables.at(executableIndex);
    if (executable.type != ExecutableKind::Graph && executable.type != ExecutableKind::Compute) {
        throw std::runtime_error("Session only supports data graph and compute shader executables");
    }

    const auto &compiledExecutable = compiledExecutionState->compiledExecutables.value().at(executableIndex);
    auto &executableState = executableStates.emplace_back(compiledExecutable);

    if (executable.type == ExecutableKind::Compute) {
        return;
    }

    const vk::DataGraphPipelineSessionCreateInfoARM sessionCreateInfo({}, *compiledExecutable.pipeline);
    executableState.graphSession = vk::raii::DataGraphPipelineSessionARM(contextView.device.get(), sessionCreateInfo);

    auto bindInfos =
        allocateDataGraphSessionMemory(contextView, executableState.graphSession, executableState.sessionMemory);
    if (!bindInfos.empty()) {
        contextView.device.get().bindDataGraphPipelineSessionMemoryARM(bindInfos);
    }
}

void Session::Impl::compileOrReuseExecutables() {
    if (compiledExecutionState->compiledExecutables.has_value()) {
        if (!moduleImplementations.empty()) {
            throw std::runtime_error("Compilation inputs cannot be rebound after compiled execution is configured");
        }
        return;
    }

    std::vector<detail::CompiledExecutable> compiledExecutables;
    compiledExecutables.reserve(workload.executableCount());
    for (uint32_t executableIndex = 0; executableIndex < workload.executableCount(); ++executableIndex) {
        auto &compiledExecutable = compiledExecutables.emplace_back();
        createPipeline(compiledExecutable, executableIndex);
    }
    compiledExecutionState->compiledExecutables.emplace(std::move(compiledExecutables));
    moduleImplementations.clear();
}

void Session::Impl::configure() {
    if (configured) {
        throw std::runtime_error("Session::configure() must only be called once");
    }

    // Compiled executables
    compileOrReuseExecutables();

    // Executable session state
    executableStates.reserve(workload.executableCount());
    for (uint32_t executableIndex = 0; executableIndex < workload.executableCount(); ++executableIndex) {
        configureExecutableState(executableIndex);
    }

    // Runtime-owned command state
    commandPool = vk::raii::CommandPool(
        contextView.device.get(), {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, contextView.queueFamilyIndex});
    commandBuffer = std::move(
        contextView.device.get().allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    fence = vk::raii::Fence(contextView.device.get(), {vk::FenceCreateFlagBits::eSignaled});
    configured = true;
}

/*******************************************************************************
 * Session lifetime
 *******************************************************************************/

Session::Session(Context &context, const Workload &workload, SessionOptions options)
    : impl_(std::make_unique<Impl>(context, workload, options)) {}

Session::Session(Context &context, const Workload &workload, CompiledExecution &compiledExecution)
    : impl_(std::make_unique<Impl>(context, workload, compiledExecution)) {}

Session::~Session() = default;

/*******************************************************************************
 * Session state access
 *******************************************************************************/

Session::Impl &Session::sessionImpl() noexcept { return *impl_; }

const Session::Impl &Session::sessionImpl() const noexcept { return *impl_; }

/*******************************************************************************
 * Session configuration
 *******************************************************************************/

void Session::bindModule(PlaceholderModuleView placeholderModule, ModuleImplementation implementation) {
    auto &sessionState = sessionImpl();
    if (sessionState.configured) {
        throw std::runtime_error("Session::bindModule() must be called before Session::configure()");
    }
    if (sessionState.compiledExecutionState->compiledExecutables.has_value()) {
        throw std::runtime_error("Compilation inputs cannot be rebound after compiled execution is configured");
    }
    if (placeholderModule.workload_ != &sessionState.workload) {
        throw std::runtime_error("PlaceholderModuleView belongs to a different Workload");
    }

    const auto moduleIndex = placeholderModule.moduleIndex_;
    const auto &placeholder = workloadImpl(sessionState.workload).modules.at(moduleIndex);
    if (placeholder.codeKind != ModuleCodeKind::Missing) {
        throw std::runtime_error("PlaceholderModuleView does not reference a missing workload module");
    }
    if (implementation.codeKind == ModuleCodeKind::Missing) {
        throw std::runtime_error("ModuleImplementation must provide executable module code");
    }

    sessionState.moduleImplementations[moduleIndex] =
        detail::moduleFromImplementation(std::move(implementation), placeholder.name, placeholder.entryPoint);
}

void Session::configure() { sessionImpl().configure(); }

/*******************************************************************************
 * Session factories
 *******************************************************************************/

BindingSet Session::createBindingSet() {
    if (!sessionImpl().configured) {
        throw std::runtime_error("Session::configure() must be called before Session::createBindingSet()");
    }
    return BindingSet(*this);
}

PreparedExecution Session::prepare(const BindingSet &bindings) {
    if (!sessionImpl().configured) {
        throw std::runtime_error("Session::configure() must be called before Session::prepare()");
    }
    if (bindings.bindingSetImpl() == nullptr) {
        throw std::runtime_error("BindingSet is invalid");
    }
    if (&bindings.session() != this) {
        throw std::runtime_error("BindingSet is associated with a different Session");
    }
    return {*this, bindings};
}

} // namespace mlsdk::workloadlib
