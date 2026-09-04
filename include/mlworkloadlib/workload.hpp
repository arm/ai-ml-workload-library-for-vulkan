/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/utils.hpp"
#include "mlworkloadlib/workload_types.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace mlworkloadlib {

class BindingSet;
class PreparedExecution;
class Session;
class Workload;

namespace detail {
class WorkloadBuilder;
} // namespace detail

/*******************************************************************************
 * Resource requirement views
 *******************************************************************************/

class TensorRequirementsView {
  public:
    vk::TensorUsageFlagsARM usage() const;
    ArrayView<const int64_t> shape() const;
    ArrayView<const int64_t> stride() const;

  private:
    TensorRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class ResourceRequirementsView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

class BufferRequirementsView {
  public:
    vk::BufferUsageFlags usage() const;

  private:
    BufferRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class ResourceRequirementsView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

class ImageRequirementsView {
  public:
    vk::Extent3D extent() const;
    vk::ImageUsageFlags usage() const;
    bool isSampled() const;
    bool isStorage() const;
    bool hasRuntimeSampler() const;
    bool requiresSamplerBinding() const;
    vk::ImageLayout requiredLayout() const;
    vk::ImageSubresourceRange requiredSubresourceRange() const;

  private:
    ImageRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class ResourceRequirementsView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

// Non-owning view of bindable workload public resource requirements.
class ResourceRequirementsView {
  public:
    ResourceKind kind() const;

    vk::DescriptorType descriptorType() const;

    vk::Format format() const;

    bool participatesInAliasing() const;

    bool requiresBoundMemoryInfo() const;

    vk::DeviceSize elementCount() const;

    vk::DeviceSize byteSize() const;

    TensorRequirementsView asTensor() const;
    BufferRequirementsView asBuffer() const;
    ImageRequirementsView asImage() const;

  private:
    ResourceRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class Workload;
    friend class ResourceView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

/*******************************************************************************
 * Resource views
 *******************************************************************************/

class ResourceView {
  public:
    uint32_t index() const;
    std::string_view name() const;

    ResourceAccess access() const;

    ResourceRequirementsView requirements() const;

  private:
    ResourceView(const Workload *workload, uint32_t index, uint32_t resourceIndex)
        : workload_(workload), index_(index), resourceIndex_(resourceIndex) {}

    friend class Workload;
    friend class BindingSet;

    const Workload *workload_ = nullptr;
    uint32_t index_ = 0;
    uint32_t resourceIndex_ = 0;
};

/*******************************************************************************
 * Module views
 *******************************************************************************/

class ModuleView {
  public:
    uint32_t index() const;

    std::string_view name() const;

    std::string_view entryPoint() const;

    ModuleCodeKind codeKind() const;

    bool requiresImplementation() const;

  private:
    ModuleView(const Workload *workload, uint32_t index) : workload_(workload), index_(index) {}

    friend class Workload;
    friend class ExecutableView;
    friend class PlaceholderModuleView;

    const Workload *workload_ = nullptr;
    uint32_t index_ = 0;
};

class PlaceholderModuleView {
  public:
    uint32_t index() const;

    ModuleView module() const;

  private:
    PlaceholderModuleView(const Workload *workload, uint32_t index, uint32_t moduleIndex)
        : workload_(workload), index_(index), moduleIndex_(moduleIndex) {}

    friend class Workload;
    friend class Session;

    const Workload *workload_ = nullptr;
    uint32_t index_ = 0;
    uint32_t moduleIndex_ = 0;
};

/*******************************************************************************
 * Executable metadata views
 *******************************************************************************/

class DataGraphPipelineMetadataView {
  public:
    std::string_view identifier() const;
    vk::PipelineCreateFlags2 flags() const;
    ArrayView<const vk::SpecializationMapEntry> specializationMapEntries() const;
    ArrayView<const uint8_t> specializationData() const;

  private:
    DataGraphPipelineMetadataView(const Workload *workload, uint32_t executableIndex)
        : workload_(workload), executableIndex_(executableIndex) {}

    friend class ExecutableView;

    const Workload *workload_ = nullptr;
    uint32_t executableIndex_ = 0;
};

/*******************************************************************************
 * Descriptor metadata
 *******************************************************************************/

struct InterfaceDescriptorBindingInfo {
    uint32_t set = 0;
    uint32_t binding = 0;
    uint32_t resourceIndex = 0;
    ResourceAccess access = ResourceAccess::ReadWrite;
    ResourceKind kind = ResourceKind::Unknown;
    vk::DescriptorType descriptorType = {};
};

/*******************************************************************************
 * Executable views
 *******************************************************************************/

class ExecutableView {
  public:
    uint32_t index() const;

    std::string_view name() const;

    ExecutableKind type() const;

    ModuleView module() const;

    DataGraphPipelineMetadataView dataGraphPipelineMetadata() const;

    uint32_t interfaceDescriptorBindingCount() const;

    InterfaceDescriptorBindingInfo interfaceDescriptorBinding(uint32_t bindingIndex) const;

  private:
    ExecutableView(const Workload *workload, uint32_t index) : workload_(workload), index_(index) {}

    friend class Workload;

    const Workload *workload_ = nullptr;
    uint32_t index_ = 0;
};

/*******************************************************************************
 * Workload
 *******************************************************************************/

// Common workload description consumed by Session. Returned view objects are
// non-owning and must not outlive the Workload.
class Workload {
  private:
    template <typename Derived> class RangeIteratorBase {
      public:
        Derived &operator++() {
            ++index_;
            return static_cast<Derived &>(*this);
        }

        bool operator==(const Derived &other) const {
            const auto &otherBase = static_cast<const RangeIteratorBase &>(other);
            return workload_ == otherBase.workload_ && index_ == otherBase.index_;
        }

        bool operator!=(const Derived &other) const { return !(*this == other); }

      protected:
        const Workload *workload() const noexcept { return workload_; }
        uint32_t index() const noexcept { return index_; }

      private:
        friend Derived;

        RangeIteratorBase(const Workload *workload, uint32_t index) : workload_(workload), index_(index) {}

        const Workload *workload_ = nullptr;
        uint32_t index_ = 0;
    };

  public:
    /***************************************************************************
     * Range helper types
     ***************************************************************************/

    class ResourceRange {
      public:
        class Iterator : public RangeIteratorBase<Iterator> {
            using Base = RangeIteratorBase<Iterator>;

          public:
            ResourceView operator*() const;

          private:
            Iterator(const Workload *workload, uint32_t publicIndex) : Base(workload, publicIndex) {}

            friend class ResourceRange;
        };

        Iterator begin() const;
        Iterator end() const;

      private:
        explicit ResourceRange(const Workload *workload) : workload_(workload) {}

        friend class Workload;

        const Workload *workload_ = nullptr;
    };

    class ExecutableRange {
      public:
        class Iterator : public RangeIteratorBase<Iterator> {
            using Base = RangeIteratorBase<Iterator>;

          public:
            ExecutableView operator*() const;

          private:
            Iterator(const Workload *workload, uint32_t index) : Base(workload, index) {}

            friend class ExecutableRange;
        };

        Iterator begin() const;
        Iterator end() const;

      private:
        explicit ExecutableRange(const Workload *workload) : workload_(workload) {}

        friend class Workload;

        const Workload *workload_ = nullptr;
    };

    class PlaceholderModuleRange {
      public:
        class Iterator : public RangeIteratorBase<Iterator> {
            using Base = RangeIteratorBase<Iterator>;

          public:
            PlaceholderModuleView operator*() const;

          private:
            Iterator(const Workload *workload, uint32_t placeholderIndex) : Base(workload, placeholderIndex) {}

            friend class PlaceholderModuleRange;
        };

        Iterator begin() const;
        Iterator end() const;

      private:
        explicit PlaceholderModuleRange(const Workload *workload) : workload_(workload) {}

        friend class Workload;

        const Workload *workload_ = nullptr;
    };

    /***************************************************************************
     * Construction
     ***************************************************************************/

    // Load a VGF-backed workload from a file.
    static Workload fromVGF(const std::filesystem::path &path);

    // Decode a VGF-backed workload from caller-owned memory.
    // Decoded constant payloads borrow from this memory for the Workload lifetime.
    static Workload fromVGF(const void *data, std::size_t size);

    // Build a workload containing one standalone compute executable.
    static Workload fromComputeShader(ComputeShaderDescription description);

    // Build a workload containing one standalone data graph executable.
    static Workload fromDataGraph(DataGraphDescription description);

    /***************************************************************************
     * Lifetime
     ***************************************************************************/

    virtual ~Workload();

    Workload(const Workload &) = delete;
    Workload &operator=(const Workload &) = delete;
    Workload(Workload &&) = delete;
    Workload &operator=(Workload &&) = delete;

    /***************************************************************************
     * Resource inspection
     ***************************************************************************/

    uint32_t resourceCount() const;

    ResourceView resource(uint32_t publicResourceIndex) const;
    ResourceRange resources() const;

    /***************************************************************************
     * Executable inspection
     ***************************************************************************/

    uint32_t executableCount() const;

    ExecutableView executable(uint32_t executableIndex) const;
    ExecutableRange executables() const;

    /***************************************************************************
     * Placeholder module inspection
     ***************************************************************************/

    uint32_t placeholderModuleCount() const;

    PlaceholderModuleView placeholderModule(uint32_t placeholderModuleIndex) const;
    PlaceholderModuleRange placeholderModules() const;

  protected:
    // Implementation type
    struct Impl;

    // Lifetime
    explicit Workload(std::unique_ptr<Impl> impl);

  private:
    // Friends
    friend class BindingSet;
    friend class Session;
    friend class PreparedExecution;

    friend class ModuleView;
    friend class PlaceholderModuleView;
    friend class ExecutableView;

    friend class ResourceView;
    friend class ResourceRequirementsView;
    friend class TensorRequirementsView;
    friend class BufferRequirementsView;
    friend class ImageRequirementsView;

    friend class detail::WorkloadBuilder;

    // Implementation access
    const Impl &workloadImpl() const noexcept;
    friend const Impl &workloadImpl(const Workload &workload) noexcept;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlworkloadlib
