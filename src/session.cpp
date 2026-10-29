/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "internal/binding_set_impl.hpp"
#include "internal/module_compiler.hpp"
#include "internal/runtime_resource.hpp"
#include "internal/session_impl.hpp"
#include "internal/session_tiling.hpp"
#include "internal/utils.hpp"
#include "internal/workload_builder.hpp"

#include "mlworkloadlib/session.hpp"

#include <vulkan/vulkan_beta.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlsdk::workloadlib {

using DataGraphBinaryInput = detail::DataGraphBinaryInput;
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
        validateSupportedDescriptorType(workloadState.descriptorTypeForBinding(descBinding));
    }
}

void validateExecutableResourceMetadata(const Workload &workload, const Executable &executable) {
    const auto &workloadState = workloadImpl(workload);
    for (const auto &descBinding : executable.bindings) {
        const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
        if (descriptorType == vk::DescriptorType::eStorageBuffer) {
            continue;
        }
        const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
        if (resource.role != detail::Resource::Role::Intermediate &&
            (executable.type == ExecutableKind::Compute ||
             descriptorType == vk::DescriptorType::eCombinedImageSampler ||
             descriptorType == vk::DescriptorType::eStorageImage)) {
            continue;
        }
        if (descriptorType == vk::DescriptorType::eCombinedImageSampler ||
            descriptorType == vk::DescriptorType::eStorageImage) {
            (void)resource.imageExtent();
            continue;
        }
        if (resource.byteSize() == 0) {
            throw std::runtime_error("Session requires non-zero resource size metadata");
        }
        if (descriptorType == vk::DescriptorType::eTensorARM) {
            detail::validateTensorMetadata(resource.format, resource.shape, resource.stride);
        }
    }
}

const std::vector<int64_t> &
effectiveTensorStrides(const Workload &workload, uint32_t resourceIndex,
                       const std::map<uint32_t, std::vector<int64_t>> &tensorStrideOverrides) {
    const auto strideOverride = tensorStrideOverrides.find(resourceIndex);
    return strideOverride == tensorStrideOverrides.end() ? workloadImpl(workload).resources.at(resourceIndex).stride
                                                         : strideOverride->second;
}

void populateDataGraphResources(DataGraphPipelineCreateStorage &storage, const Executable &executable,
                                const Workload &workload, const std::vector<detail::ResourceTiling> &resourceTilings,
                                const std::map<uint32_t, std::vector<int64_t>> &tensorStrideOverrides) {
    const auto &workloadState = workloadImpl(workload);
    storage.resourceTensorDescriptions.reserve(executable.bindings.size());
    storage.imageLayouts.reserve(executable.bindings.size());
    storage.resourceInfos.reserve(executable.bindings.size());
    for (const auto &descBinding : executable.bindings) {
        const auto &resource = workloadState.resources.at(descBinding.resourceIndex);
        const auto descriptorType = workloadState.descriptorTypeForBinding(descBinding);
        if (descriptorType == vk::DescriptorType::eCombinedImageSampler ||
            descriptorType == vk::DescriptorType::eStorageImage) {
            storage.imageLayouts.emplace_back(resource.requiredImageLayout());
            storage.resourceInfos.emplace_back(descBinding.set, descBinding.binding, 0, &storage.imageLayouts.back());
        } else {
            const auto tensorTiling = detail::toTensorTiling(resourceTilings.at(descBinding.resourceIndex));
            const auto &strides =
                descriptorType == vk::DescriptorType::eTensorARM
                    ? effectiveTensorStrides(workload, descBinding.resourceIndex, tensorStrideOverrides)
                    : resource.stride;
            if (descriptorType == vk::DescriptorType::eTensorARM) {
                const auto description =
                    detail::tensorDescription(workload, descBinding.resourceIndex, tensorTiling, strides);
                detail::validateTensorMetadata(resource.format, resource.shape, strides);
                detail::validateTensorTiling(tensorTiling, description.usage, resource.shape, !strides.empty());
                if (!(description.usage & vk::TensorUsageFlagBitsARM::eDataGraph)) {
                    throw std::runtime_error("Graph tensor resources require eDataGraph usage");
                }
                storage.resourceTensorDescriptions.push_back(description);
            } else {
                storage.resourceTensorDescriptions.emplace_back(
                    tensorTiling, resource.format, static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                    strides.empty() ? nullptr : strides.data(), vk::TensorUsageFlagBitsARM::eDataGraph);
            }
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
        detail::validateTensorTiling(vk::TensorTilingARM::eLinear, detail::tensorMetadata(resource).usage,
                                     resource.shape, !resource.stride.empty());
        detail::validateTensorMetadata(resource.format, resource.shape, resource.stride);
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
    const auto maxDescriptorSets = contextView.physicalDevice.get().getProperties().limits.maxBoundDescriptorSets;
    const auto descBindingSets = detail::splitBindingsBySet(descBindings, maxDescriptorSets);

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
                                         const vk::raii::PipelineCache *pipelineCache,
                                         const std::optional<PipelineRobustnessOptions> &pipelineRobustness) {
    const auto &dispatchShape = executable.dispatchShape;
    if (dispatchShape[0] == 0 || dispatchShape[1] == 0 || dispatchShape[2] == 0) {
        throw std::runtime_error("Compute shader executables must have a non-zero 3D dispatch shape");
    }

    SpecializationInfoStorage specializationInfo;
    const vk::PipelineShaderStageCreateInfo shaderStageCreateInfo(
        {}, vk::ShaderStageFlagBits::eCompute, *shaderModule, compiledModule.entryPoint.c_str(),
        specializationInfo.populate(executable.specializationInfo));
    vk::ComputePipelineCreateInfo pipelineCreateInfo({}, shaderStageCreateInfo, *pipelineLayout);
    vk::PipelineRobustnessCreateInfo pipelineRobustnessInfo;
    if (pipelineRobustness.has_value()) {
        pipelineRobustnessInfo.storageBuffers = pipelineRobustness->storageBuffers;
        pipelineRobustnessInfo.uniformBuffers = pipelineRobustness->uniformBuffers;
        pipelineRobustnessInfo.vertexInputs = pipelineRobustness->vertexInputs;
        pipelineRobustnessInfo.images = pipelineRobustness->images;
        detail::insertPNextAfter(pipelineCreateInfo, pipelineRobustnessInfo);
    }
    return {contextView.device.get(), pipelineCache, pipelineCreateInfo};
}

vk::raii::Pipeline createDataGraphPipeline(const Executable &executable, const Module &compiledModule,
                                           const vk::raii::ShaderModule &shaderModule,
                                           const vk::raii::PipelineLayout &pipelineLayout, const Workload &workload,
                                           const ContextView &contextView, const vk::raii::PipelineCache *pipelineCache,
                                           bool enableNeuralStatistics,
                                           const std::vector<detail::ResourceTiling> &resourceTilings,
                                           const std::map<uint32_t, std::vector<int64_t>> &tensorStrideOverrides) {
    if (enableNeuralStatistics && (executable.dataGraphPipelineFlags &
                                   vk::PipelineCreateFlagBits2::eProtectedAccessOnly) != vk::PipelineCreateFlags2{}) {
        throw std::runtime_error("Neural statistics cannot be enabled for a protected-access-only data graph pipeline");
    }

    DataGraphPipelineCreateStorage storage;
    populateDataGraphResources(storage, executable, workload, resourceTilings, tensorStrideOverrides);
    populateDataGraphConstants(storage, executable, workload);

    SpecializationInfoStorage specializationInfo;
    vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleInfo(
        *shaderModule, compiledModule.entryPoint.c_str(), specializationInfo.populate(executable.specializationInfo),
        static_cast<uint32_t>(storage.constants.size()), storage.constants.data(), nullptr);
    const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo(executable.dataGraphPipelineFlags, *pipelineLayout,
                                                                static_cast<uint32_t>(storage.resourceInfos.size()),
                                                                storage.resourceInfos.data(), &shaderModuleInfo);
    vk::DataGraphPipelineNeuralStatisticsCreateInfoARM neuralStatisticsInfo(enableNeuralStatistics);
    if (enableNeuralStatistics) {
        detail::insertPNextAfter(shaderModuleInfo, neuralStatisticsInfo);
    }
    const vk::raii::DeferredOperationKHR deferredOperation(nullptr);
    return {contextView.device.get(), deferredOperation, pipelineCache, pipelineCreateInfo};
}

#ifdef VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME
vk::raii::Pipeline createDataGraphBinaryPipeline(const Executable &executable,
                                                 const vk::raii::PipelineLayout &pipelineLayout,
                                                 const ContextView &contextView, const DataGraphBinaryInput &input,
                                                 bool enableNeuralStatistics) {
    const auto properties =
        contextView.physicalDevice.get()
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDataGraphBinariesPropertiesARM>();
    const auto alignment =
        properties.get<vk::PhysicalDeviceDataGraphBinariesPropertiesARM>().dataGraphPipelineBinaryDataAlignment;
    if (alignment == 0 || alignment > std::numeric_limits<std::size_t>::max() || (alignment & (alignment - 1)) != 0) {
        throw std::runtime_error("Invalid data graph binary alignment reported by the physical device");
    }
    if (reinterpret_cast<std::uintptr_t>(input.bytes.data()) % alignment != 0) {
        throw std::runtime_error("Data graph binary is not aligned for the physical device");
    }

    if (enableNeuralStatistics && (executable.dataGraphPipelineFlags &
                                   vk::PipelineCreateFlagBits2::eProtectedAccessOnly) != vk::PipelineCreateFlags2{}) {
        throw std::runtime_error("Neural statistics cannot be enabled for a protected-access-only data graph pipeline");
    }

    const bool useApplicationStorage = input.storageMode == DataGraphBinaryStorageMode::ApplicationBacked;
    vk::DataGraphPipelineBinaryCreateInfoARM binaryInfo(useApplicationStorage, input.bytes.size(), input.bytes.data());
    vk::DataGraphPipelineIdentifierCreateInfoARM identifierInfo;
    if (!input.identifier.empty()) {
        identifierInfo.identifierSize = static_cast<uint32_t>(input.identifier.size());
        identifierInfo.pIdentifier = input.identifier.data();
        detail::insertPNextAfter(binaryInfo, identifierInfo);
    }
    vk::DataGraphPipelineNeuralStatisticsCreateInfoARM neuralStatisticsInfo(enableNeuralStatistics);
    if (enableNeuralStatistics) {
        detail::insertPNextAfter(binaryInfo, neuralStatisticsInfo);
    }

    const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo(executable.dataGraphPipelineFlags, *pipelineLayout, 0,
                                                                nullptr, &binaryInfo);
    const vk::raii::DeferredOperationKHR deferredOperation(nullptr);
    vk::raii::Pipeline pipeline(contextView.device.get(), deferredOperation, nullptr, pipelineCreateInfo);
    const auto creationResult = pipeline.getConstructorSuccessCode();
    if (creationResult != vk::Result::eSuccess) {
        throw std::runtime_error("Data graph binary pipeline creation returned " + vk::to_string(creationResult));
    }
    if (!*pipeline) {
        throw std::runtime_error("Data graph binary pipeline creation returned a null pipeline");
    }
    return pipeline;
}
#endif

/*******************************************************************************
 * Data graph session memory
 *******************************************************************************/

std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> allocateDataGraphSessionMemory(
    const ContextView &contextView, const vk::raii::DataGraphPipelineSessionARM &graphSession,
    std::vector<vk::raii::DeviceMemory> &sessionMemory, std::vector<DataGraphSessionMemoryInfo> &sessionMemoryInfo,
    const SessionOptions &options) {
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

            const bool neuralStatistics = bindPointRequirement.bindPoint ==
                                          vk::DataGraphPipelineSessionBindPointARM::eNeuralAcceleratorStatistics;
            const vk::MemoryPropertyFlags defaultMemoryProperties =
                options.neuralStatistics.has_value() && neuralStatistics
                    ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
                    : vk::MemoryPropertyFlagBits::eDeviceLocal;
            const auto memoryProperties = options.requiredSessionMemoryProperties.value_or(defaultMemoryProperties);
            const auto memoryType = detail::findMemoryType(contextView.physicalDevice.get(),
                                                           memReqs.memoryRequirements.memoryTypeBits, memoryProperties);
            const auto selectedMemoryProperties =
                contextView.physicalDevice.get().getMemoryProperties().memoryTypes[memoryType].propertyFlags;
            const vk::MemoryAllocateInfo allocateInfo(memReqs.memoryRequirements.size, memoryType);
            const auto &deviceMemory = sessionMemory.emplace_back(contextView.device.get(), allocateInfo);
            sessionMemoryInfo.push_back({*deviceMemory, memReqs.memoryRequirements.size, bindPointRequirement.bindPoint,
                                         objectIndex, selectedMemoryProperties, neuralStatistics});
            bindInfos.emplace_back(*graphSession, bindPointRequirement.bindPoint, objectIndex, *deviceMemory);
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
      compiledExecutionState(std::make_shared<CompiledExecution::Impl>(contextIn, workloadIn, optionsIn)),
      resourceTilings(detail::defaultResourceTilings(workloadIn)) {}

Session::Impl::Impl(Context &contextIn, const Workload &workloadIn, CompiledExecution &compiledExecutionIn)
    : workload(workloadIn), contextView(contextIn.contextImpl().contextView()),
      compiledExecutionState(compiledExecutionIn.compiledExecutionImpl()) {
    if (compiledExecutionState->context != &contextIn) {
        throw std::runtime_error("CompiledExecution was not created for this Context");
    }
    if (compiledExecutionState->workload != &workloadIn) {
        throw std::runtime_error("CompiledExecution was not created for this Workload");
    }
    resourceTilings = detail::defaultResourceTilings(workloadIn);
    if (compiledExecutionState->compiledState.has_value()) {
        const auto &pipelineTilings = compiledExecutionState->compiledState->pipelineTilings;
        for (std::size_t index = 0; index < pipelineTilings.size(); ++index) {
            if (pipelineTilings[index].has_value()) {
                resourceTilings.at(index) = *pipelineTilings[index];
            }
        }
    }
}

void Session::Impl::selectTiling(uint32_t resourceIndex, detail::ResourceTiling tiling) {
    if (compiledExecutionState->compiledState.has_value()) {
        // Pipeline tilings include every peer of a graph-tensor alias group.
        const auto &compiledTiling = compiledExecutionState->compiledState->pipelineTilings.at(resourceIndex);
        if (compiledTiling.has_value() && tiling != *compiledTiling) {
            throw std::runtime_error("Resource " + std::to_string(resourceIndex) +
                                     " tiling differs from the compiled data graph pipeline");
        }
    }
    auto selectedTilings = resourceTilings;
    detail::selectResourceTiling(workload, selectedTilings, resourceIndex, tiling);
    for (const auto &[index, strides] : tensorStrideOverrides) {
        const auto tensorTiling = detail::toTensorTiling(selectedTilings.at(index));
        detail::validateTensorTiling(tensorTiling, detail::tensorUsageForTiling(workload, index, tensorTiling),
                                     workloadImpl(workload).resources.at(index).shape, !strides.empty());
    }
    resourceTilings.swap(selectedTilings);
}

/*******************************************************************************
 * Compiled workload construction
 *******************************************************************************/

void Session::Impl::createPipeline(detail::CompiledExecutable &compiledExecutable, uint32_t executableIndex) const {
    const auto &executable = workloadImpl(workload).executables.at(executableIndex);
    if (executable.type != ExecutableKind::Graph && executable.type != ExecutableKind::Compute) {
        throw std::runtime_error("Session only supports data graph and compute shader executables");
    }

    validateExecutableBindings(workload, executable);
    validateExecutableResourceMetadata(workload, executable);

    // Pipeline layout
    compiledExecutable.pushConstantRanges = executable.pushConstantRanges;
    compiledExecutable.descriptorSetLayouts = createDescriptorSetLayouts(executable.bindings, workload, contextView);
    compiledExecutable.pipelineLayout =
        createPipelineLayout(compiledExecutable.descriptorSetLayouts, executable, contextView);
    const auto &options = compiledExecutionState->options;

#ifdef VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME
    if (const auto binaryInput = dataGraphBinaryInputs.find(executableIndex);
        executable.type == ExecutableKind::Graph && binaryInput != dataGraphBinaryInputs.end()) {
        if (std::any_of(executable.bindings.begin(), executable.bindings.end(),
                        [&](const auto &binding) { return tensorStrideOverrides.count(binding.resourceIndex) != 0; })) {
            throw std::runtime_error("Tensor stride overrides are not supported by data graph binaries");
        }
        compiledExecutable.pipeline =
            createDataGraphBinaryPipeline(executable, compiledExecutable.pipelineLayout, contextView,
                                          binaryInput->second, options.neuralStatistics.has_value());
        return;
    }
#endif

    if (executable.type == ExecutableKind::Compute) {
        auto compiledModule = moduleForExecutable(workload, moduleImplementations, executableIndex);
        detail::compileModuleToSpirv(compiledModule, executable.type);
        compiledExecutable.shaderModule = createShaderModule(compiledModule, contextView);
        compiledExecutable.pipeline = createComputePipeline(executable, compiledModule, compiledExecutable.shaderModule,
                                                            compiledExecutable.pipelineLayout, contextView,
                                                            options.pipelineCache, options.pipelineRobustness);
    } else {
        auto compiledModule = moduleForExecutable(workload, moduleImplementations, executableIndex);
        detail::compileModuleToSpirv(compiledModule, executable.type);
        compiledExecutable.shaderModule = createShaderModule(compiledModule, contextView);
        compiledExecutable.pipeline =
            createDataGraphPipeline(executable, compiledModule, compiledExecutable.shaderModule,
                                    compiledExecutable.pipelineLayout, workload, contextView, options.pipelineCache,
                                    options.neuralStatistics.has_value(), resourceTilings, tensorStrideOverrides);
    }
}

/*******************************************************************************
 * Configuration
 *******************************************************************************/

void Session::Impl::configureExecutableState(uint32_t executableIndex) {
    const auto &executable = workloadImpl(workload).executables.at(executableIndex);
    if (executable.type != ExecutableKind::Graph && executable.type != ExecutableKind::Compute) {
        throw std::runtime_error("Session only supports data graph and compute shader executables");
    }

    const auto &compiledExecutable = compiledExecutionState->compiledState->executables.at(executableIndex);
    auto &executableState = executableStates.emplace_back(compiledExecutable);

    if (executable.type == ExecutableKind::Compute) {
        return;
    }

    vk::DataGraphPipelineSessionCreateInfoARM sessionCreateInfo({}, *compiledExecutable.pipeline);
    vk::DataGraphPipelineSessionNeuralStatisticsCreateInfoARM neuralStatisticsInfo;
    if (compiledExecutionState->options.neuralStatistics.has_value()) {
        neuralStatisticsInfo.mode = compiledExecutionState->options.neuralStatistics->mode;
        detail::insertPNextAfter(sessionCreateInfo, neuralStatisticsInfo);
    }
    executableState.graphSession = vk::raii::DataGraphPipelineSessionARM(contextView.device.get(), sessionCreateInfo);

    auto bindInfos =
        allocateDataGraphSessionMemory(contextView, executableState.graphSession, executableState.sessionMemory,
                                       executableState.sessionMemoryInfo, compiledExecutionState->options);
    if (!bindInfos.empty()) {
        contextView.device.get().bindDataGraphPipelineSessionMemoryARM(bindInfos);
    }
}

const Session::Impl::ExecutableState &Session::Impl::graphExecutableState(uint32_t executableIndex) const {
    if (!configured) {
        throw std::runtime_error("Session::configure() must be called before querying graph diagnostics");
    }
    if (executableIndex >= executableStates.size() ||
        workloadImpl(workload).executables.at(executableIndex).type != ExecutableKind::Graph) {
        throw std::runtime_error("Executable index does not reference a configured graph executable");
    }
    return executableStates[executableIndex];
}

std::vector<vk::DataGraphPipelinePropertyARM>
Session::Impl::dataGraphPipelineProperties(uint32_t executableIndex) const {
    const auto &pipeline = graphExecutableState(executableIndex).compiledExecutable.get().pipeline;
    return contextView.device.get().getDataGraphPipelineAvailablePropertiesARM(vk::DataGraphPipelineInfoARM(*pipeline));
}

DataGraphPipelinePropertyData
Session::Impl::dataGraphPipelineProperty(uint32_t executableIndex, vk::DataGraphPipelinePropertyARM property) const {
    const auto availableProperties = dataGraphPipelineProperties(executableIndex);
    if (std::find(availableProperties.begin(), availableProperties.end(), property) == availableProperties.end()) {
        throw std::runtime_error("Requested graph-pipeline property is not available");
    }

    const auto &pipeline = graphExecutableState(executableIndex).compiledExecutable.get().pipeline;
    const vk::DataGraphPipelineInfoARM pipelineInfo(*pipeline);
    vk::DataGraphPipelinePropertyQueryResultARM query(property);
    auto result = contextView.device.get().getDataGraphPipelinePropertiesARM(&pipelineInfo, 1, &query);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to query graph-pipeline property size");
    }

    DataGraphPipelinePropertyData propertyData{property, query.isText == vk::True, {}};
    propertyData.data.resize(query.dataSize);
    if (propertyData.data.empty()) {
        return propertyData;
    }

    query.pData = propertyData.data.data();
    result = contextView.device.get().getDataGraphPipelinePropertiesARM(&pipelineInfo, 1, &query);
    if (result == vk::Result::eIncomplete) {
        throw std::runtime_error("Graph-pipeline property data changed size while being queried");
    }
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to query graph-pipeline property data");
    }
    propertyData.isText = query.isText == vk::True;
    propertyData.data.resize(query.dataSize);
    return propertyData;
}

uint32_t Session::Impl::dataGraphSessionMemoryCount(uint32_t executableIndex) const {
    return static_cast<uint32_t>(graphExecutableState(executableIndex).sessionMemoryInfo.size());
}

DataGraphSessionMemoryInfo Session::Impl::dataGraphSessionMemory(uint32_t executableIndex, uint32_t memoryIndex) const {
    const auto &memoryInfo = graphExecutableState(executableIndex).sessionMemoryInfo;
    if (memoryIndex >= memoryInfo.size()) {
        throw std::runtime_error("Graph-session memory index is invalid");
    }
    return memoryInfo[memoryIndex];
}

void Session::Impl::compileOrReuseExecutables() {
    std::map<uint32_t, std::vector<int64_t>> aliasTensorStrides;
    const auto &workloadState = workloadImpl(workload);
    for (const auto &executable : workloadState.executables) {
        if (executable.type != ExecutableKind::Graph) {
            continue;
        }
        for (const auto &binding : executable.bindings) {
            const auto &resource = workloadState.resources.at(binding.resourceIndex);
            if (resource.role == detail::Resource::Role::Intermediate && resource.aliasGroupId.has_value() &&
                resource.descriptorType == vk::DescriptorType::eTensorARM) {
                aliasTensorStrides.emplace(binding.resourceIndex, tensorStrides(binding.resourceIndex));
            }
        }
    }
    if (compiledExecutionState->compiledState.has_value()) {
        if (!moduleImplementations.empty() || !dataGraphBinaryInputs.empty()) {
            throw std::runtime_error("Compilation inputs cannot be rebound after compiled execution is configured");
        }
        detail::validateCompiledTilings(resourceTilings, compiledExecutionState->compiledState->pipelineTilings);
        if (aliasTensorStrides != compiledExecutionState->compiledState->aliasTensorStrides) {
            throw std::runtime_error("Session alias tensor strides do not match the compiled execution");
        }
        return;
    }

    std::vector<detail::CompiledExecutable> compiledExecutables;
    compiledExecutables.reserve(workload.executableCount());
    for (uint32_t executableIndex = 0; executableIndex < workload.executableCount(); ++executableIndex) {
        auto &compiledExecutable = compiledExecutables.emplace_back();
        createPipeline(compiledExecutable, executableIndex);
    }
    compiledExecutionState->compiledState.emplace(
        detail::CompiledState{detail::fixedPipelineTilings(workload, resourceTilings), std::move(compiledExecutables),
                              std::move(aliasTensorStrides)});
    moduleImplementations.clear();
    dataGraphBinaryInputs.clear();
}

void Session::Impl::configure() {
    if (configured) {
        throw std::runtime_error("Session::configure() must only be called once");
    }

    detail::validateAliasGroupTilings(workload, resourceTilings);
    detail::validateAliasImageTensorLayouts(workload, resourceTilings);
    std::optional<vk::PhysicalDeviceTensorPropertiesARM> tensorLimits;
    const auto &resources = workloadImpl(workload).resources;
    for (size_t index = 0; index < resources.size(); ++index) {
        const auto &resource = resources[index];
        if (resource.aliasGroupId.has_value() && resource.role != detail::Resource::Role::Constant &&
            std::holds_alternative<detail::Resource::TensorMetadata>(resource.metadata)) {
            if (!tensorLimits.has_value()) {
                tensorLimits =
                    contextView.physicalDevice.get()
                        .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceTensorPropertiesARM>()
                        .get<vk::PhysicalDeviceTensorPropertiesARM>();
            }
            detail::validateAliasTensorDescription(static_cast<uint32_t>(index), resource,
                                                   tensorStrides(static_cast<uint32_t>(index)), &*tensorLimits);
        }
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

const std::vector<int64_t> &Session::Impl::tensorStrides(uint32_t resourceIndex) const {
    return effectiveTensorStrides(workload, resourceIndex, tensorStrideOverrides);
}

/*******************************************************************************
 * Session configuration
 *******************************************************************************/

void Session::bindModule(PlaceholderModuleView placeholderModule, ModuleImplementation implementation) {
    auto &sessionState = sessionImpl();
    if (sessionState.configured) {
        throw std::runtime_error("Session::bindModule() must be called before Session::configure()");
    }
    if (sessionState.compiledExecutionState->compiledState.has_value()) {
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
    const auto &executables = workloadImpl(sessionState.workload).executables;
    const auto usesBinary = [&](const auto &binaryInput) {
        return executables.at(binaryInput.first).moduleIndex == moduleIndex;
    };
    if (std::any_of(sessionState.dataGraphBinaryInputs.begin(), sessionState.dataGraphBinaryInputs.end(), usesBinary)) {
        throw std::runtime_error("A placeholder module and data graph binary cannot target the same module");
    }

    sessionState.moduleImplementations[moduleIndex] =
        detail::moduleFromImplementation(std::move(implementation), placeholder.name, placeholder.entryPoint);
}

void Session::bindDataGraphBinary(ArrayView<const std::byte> binary, DataGraphBinaryStorageMode storageMode,
                                  ArrayView<const uint8_t> pipelineIdentifier, uint32_t executableIndex) {
#ifdef VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME
    auto &sessionState = sessionImpl();
    if (sessionState.configured) {
        throw std::runtime_error("Session::bindDataGraphBinary() must be called before Session::configure()");
    }
    if (sessionState.compiledExecutionState->compiledState.has_value()) {
        throw std::runtime_error("Compilation inputs cannot be rebound after compiled execution is configured");
    }
    const auto &workloadState = workloadImpl(sessionState.workload);
    if (executableIndex >= workloadState.executables.size()) {
        throw std::runtime_error("Data graph executable index is out of range");
    }
    const auto &workloadExecutable = workloadState.executables.at(executableIndex);
    if (workloadExecutable.type != ExecutableKind::Graph) {
        throw std::runtime_error("Data graph binary requires a data graph executable");
    }
    if (sessionState.moduleImplementations.count(workloadExecutable.moduleIndex) != 0) {
        throw std::runtime_error("A placeholder module and data graph binary cannot target the same module");
    }
    if (binary.data() == nullptr || binary.empty()) {
        throw std::runtime_error("Data graph binary payload must not be empty");
    }
    if (storageMode != DataGraphBinaryStorageMode::ImplementationManaged &&
        storageMode != DataGraphBinaryStorageMode::ApplicationBacked) {
        throw std::runtime_error("Invalid data graph binary storage mode");
    }
    if (pipelineIdentifier.size() > std::numeric_limits<uint32_t>::max() ||
        (!pipelineIdentifier.empty() && pipelineIdentifier.data() == nullptr)) {
        throw std::runtime_error("Data graph pipeline identifier is invalid");
    }
    if (!workloadExecutable.constantIndexes.empty()) {
        throw std::runtime_error("Data graph binary executables cannot declare graph constants");
    }
    if (!workloadExecutable.specializationInfo.empty()) {
        throw std::runtime_error("Data graph binary executables cannot declare specialization constants");
    }
    constexpr std::size_t headerSize = 16;
    if (binary.size() < headerSize) {
        throw std::runtime_error("Data graph binary header is truncated");
    }

    const auto readLittleEndian = [](const std::byte *bytes, std::size_t count) {
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < count; ++index) {
            value |= static_cast<std::uint64_t>(std::to_integer<uint8_t>(bytes[index])) << (index * 8);
        }
        return value;
    };
    if (readLittleEndian(binary.data(), 4) != 0x1122DDFFU) {
        throw std::runtime_error("Data graph binary header has an invalid magic value");
    }
    if (readLittleEndian(binary.data() + 8, 8) != binary.size() - headerSize) {
        throw std::runtime_error("Data graph binary payload size does not match its header");
    }

    DataGraphBinaryInput input;
    input.bytes = binary;
    input.storageMode = storageMode;
    if (!pipelineIdentifier.empty()) {
        input.identifier.assign(pipelineIdentifier.begin(), pipelineIdentifier.end());
    }
    sessionState.dataGraphBinaryInputs[executableIndex] = std::move(input);
#else
    static_cast<void>(executableIndex);
    static_cast<void>(binary);
    static_cast<void>(storageMode);
    static_cast<void>(pipelineIdentifier);
    throw std::runtime_error("Vulkan headers do not support VK_ARM_data_graph_binaries");
#endif
}

void Session::validateResourceAccess(const Impl &sessionState, const ResourceView &resource) {
    if (resource.workload_ != &sessionState.workload) {
        throw std::runtime_error("ResourceView belongs to a different Workload");
    }
}

void Session::validateResourceAccessForSet(const Impl &sessionState, const ResourceView &resource) {
    if (sessionState.configured) {
        throw std::runtime_error("Resource tiling must be set before Session::configure()");
    }

    validateResourceAccess(sessionState, resource);
}

void Session::setTensorTiling(ResourceView resource, vk::TensorTilingARM tiling) {
    auto &sessionState = sessionImpl();
    validateResourceAccessForSet(sessionState, resource);
    const auto &workloadResource = workloadImpl(sessionState.workload).resources.at(resource.resourceIndex_);
    if (!std::holds_alternative<detail::Resource::TensorMetadata>(workloadResource.metadata)) {
        throw std::runtime_error("Session::setTensorTiling() requires a tensor resource");
    }
    const auto usage = detail::tensorUsageForTiling(sessionState.workload, resource.resourceIndex_, tiling);
    detail::validateTensorTiling(tiling, usage, workloadResource.shape, !workloadResource.stride.empty());
    sessionState.selectTiling(resource.resourceIndex_, detail::toResourceTiling(tiling));
}

void Session::setImageTiling(ResourceView resource, vk::ImageTiling tiling) {
    auto &sessionState = sessionImpl();
    validateResourceAccessForSet(sessionState, resource);
    const auto &workloadResource = workloadImpl(sessionState.workload).resources.at(resource.resourceIndex_);
    if (!std::holds_alternative<detail::Resource::ImageMetadata>(workloadResource.metadata)) {
        throw std::runtime_error("Session::setImageTiling() requires an image resource");
    }
    detail::validateImageTiling(tiling, resource.requirements().asImage().usage(tiling));
    sessionState.selectTiling(resource.resourceIndex_, detail::toResourceTiling(tiling));
}

vk::TensorTilingARM Session::tensorTiling(ResourceView resource) const {
    const auto &sessionState = sessionImpl();
    validateResourceAccess(sessionState, resource);
    const auto &workloadResource = workloadImpl(sessionState.workload).resources.at(resource.resourceIndex_);
    if (!std::holds_alternative<detail::Resource::TensorMetadata>(workloadResource.metadata)) {
        throw std::runtime_error("Session::tensorTiling() requires a tensor resource");
    }
    return detail::toTensorTiling(sessionState.resourceTilings.at(resource.resourceIndex_));
}

vk::ImageTiling Session::imageTiling(ResourceView resource) const {
    const auto &sessionState = sessionImpl();
    validateResourceAccess(sessionState, resource);
    const auto &workloadResource = workloadImpl(sessionState.workload).resources.at(resource.resourceIndex_);
    if (!std::holds_alternative<detail::Resource::ImageMetadata>(workloadResource.metadata)) {
        throw std::runtime_error("Session::imageTiling() requires an image resource");
    }
    return detail::toImageTiling(sessionState.resourceTilings.at(resource.resourceIndex_));
}

void Session::setTensorStrides(AliasPeerView peer, std::vector<int64_t> strides) {
    auto &sessionState = sessionImpl();
    if (sessionState.configured) {
        throw std::runtime_error("Session::setTensorStrides() must be called before Session::configure()");
    }
    if (peer.workload_ != &sessionState.workload) {
        throw std::runtime_error("AliasPeerView belongs to a different Workload");
    }
    const auto &resource = workloadImpl(sessionState.workload).resources.at(peer.resourceIndex_);
    const auto fail = [index = peer.resourceIndex_](std::string_view message) {
        throw std::runtime_error("Alias resource " + std::to_string(index) + ": " + std::string(message));
    };
    if (resource.role != detail::Resource::Role::Intermediate || !resource.aliasGroupId.has_value() ||
        resource.descriptorType != vk::DescriptorType::eTensorARM) {
        fail("tensor strides require an aliased intermediate tensor");
    }
    const auto tiling = detail::toTensorTiling(sessionState.resourceTilings.at(peer.resourceIndex_));
    if (tiling != vk::TensorTilingARM::eLinear) {
        fail("only linear tensor tiling accepts strides");
    }
    if (strides.empty()) {
        fail("tensor stride count must match its non-empty shape");
    }
    detail::validateAliasTensorDescription(peer.resourceIndex_, resource, strides);
    if (sessionState.compiledExecutionState->compiledState.has_value()) {
        const auto &fixedStrides = sessionState.compiledExecutionState->compiledState->aliasTensorStrides;
        const auto compiled = fixedStrides.find(peer.resourceIndex_);
        if (compiled != fixedStrides.end() && compiled->second != strides) {
            fail("tensor strides differ from the compiled data graph pipeline");
        }
    }
    sessionState.tensorStrideOverrides.insert_or_assign(peer.resourceIndex_, std::move(strides));
}

AliasPeerMemoryRequirements Session::aliasPeerMemoryRequirements(AliasPeerView peer) const {
    const auto &state = sessionImpl();
    if (!state.configured) {
        throw std::runtime_error("Session::configure() must be called before querying alias peer memory requirements");
    }
    if (peer.workload_ != &state.workload) {
        throw std::runtime_error("AliasPeerView belongs to a different Workload");
    }
    const auto &resource = workloadImpl(state.workload).resources.at(peer.resourceIndex_);
    if (resource.role != detail::Resource::Role::Intermediate || !resource.aliasGroupId.has_value()) {
        throw std::runtime_error("Memory requirements require an intermediate alias peer");
    }
    return detail::queryAliasPeerMemoryRequirements(state.workload, state.contextView, peer.resourceIndex_,
                                                    state.resourceTilings.at(peer.resourceIndex_),
                                                    state.tensorStrides(peer.resourceIndex_));
}

void Session::configure() { sessionImpl().configure(); }

/*******************************************************************************
 * Graph diagnostics
 *******************************************************************************/

std::vector<vk::DataGraphPipelinePropertyARM> Session::dataGraphPipelineProperties(uint32_t executableIndex) const {
    return sessionImpl().dataGraphPipelineProperties(executableIndex);
}

DataGraphPipelinePropertyData Session::dataGraphPipelineProperty(uint32_t executableIndex,
                                                                 vk::DataGraphPipelinePropertyARM property) const {
    return sessionImpl().dataGraphPipelineProperty(executableIndex, property);
}

uint32_t Session::dataGraphSessionMemoryCount(uint32_t executableIndex) const {
    return sessionImpl().dataGraphSessionMemoryCount(executableIndex);
}

DataGraphSessionMemoryInfo Session::dataGraphSessionMemory(uint32_t executableIndex, uint32_t memoryIndex) const {
    return sessionImpl().dataGraphSessionMemory(executableIndex, memoryIndex);
}

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
