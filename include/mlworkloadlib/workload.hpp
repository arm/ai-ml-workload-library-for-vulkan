/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/array_view.hpp"
#include "mlworkloadlib/workload_types.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace mlsdk::workloadlib {

class BindingSet;
class PreparedExecution;
class Session;
class Workload;

/** @cond */
namespace detail {
class WorkloadBuilder;
} // namespace detail
/** @endcond */

/*******************************************************************************
 * Resource requirement views
 *******************************************************************************/

/** @brief Non-owning view of tensor-specific workload requirements. */
class TensorRequirementsView {
  public:
    /** @brief Returns the Vulkan tensor usage flags required by the workload. */
    vk::TensorUsageFlagsARM usage() const;

    /** @brief Returns a borrowed view of the tensor dimensions. */
    ArrayView<const int64_t> shape() const;

    /** @brief Returns a borrowed view of optional Vulkan tensor stride metadata. */
    ArrayView<const int64_t> stride() const;

  private:
    TensorRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class ResourceRequirementsView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

/** @brief Non-owning view of storage-buffer-specific workload requirements. */
class BufferRequirementsView {
  public:
    /** @brief Returns the Vulkan buffer usage flags required by the workload. */
    vk::BufferUsageFlags usage() const;

  private:
    BufferRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class ResourceRequirementsView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

/** @brief Non-owning view of image-specific workload requirements. */
class ImageRequirementsView {
  public:
    /** @brief Returns the required image extent. */
    vk::Extent3D extent() const;

    /** @brief Returns the required Vulkan image usage flags. */
    vk::ImageUsageFlags usage() const;

    /** @brief Returns true for a combined image-sampler descriptor. */
    bool isSampled() const;

    /** @brief Returns true for a storage-image descriptor. */
    bool isStorage() const;

    /** @brief Returns true if the runtime must create the sampler from workload metadata. */
    bool hasRuntimeSampler() const;

    /** @brief Returns true if the caller must provide a sampler in ImageBindingInfo. */
    bool requiresSamplerBinding() const;

    /** @brief Returns the image layout required when the resource is accessed. */
    vk::ImageLayout requiredLayout() const;

    /** @brief Returns the image subresource range required by the workload. */
    vk::ImageSubresourceRange requiredSubresourceRange() const;

  private:
    ImageRequirementsView(const Workload *workload, uint32_t resourceIndex)
        : workload_(workload), resourceIndex_(resourceIndex) {}

    friend class ResourceRequirementsView;

    const Workload *workload_ = nullptr;
    uint32_t resourceIndex_ = 0;
};

/**
 * @brief Non-owning view of requirements for a public workload resource.
 *
 * Use the asTensor(), asBuffer(), or asImage() member matching kind() for
 * kind-specific requirements.
 */
class ResourceRequirementsView {
  public:
    /** @brief Returns the public resource category. */
    ResourceKind kind() const;

    /** @brief Returns the Vulkan descriptor type, or a default value when none is defined. */
    vk::DescriptorType descriptorType() const;

    /** @brief Returns the resource element or texel format. */
    vk::Format format() const;

    /** @brief Returns true if the resource belongs to a Vulkan alias group. */
    bool participatesInAliasing() const;

    /** @brief Returns true if a binding must include valid BoundMemoryInfo. */
    bool requiresBoundMemoryInfo() const;

    /** @brief Returns the logical element count, or zero when it cannot be determined. */
    vk::DeviceSize elementCount() const;

    /** @brief Returns the required storage size in bytes, or zero when it cannot be determined. */
    vk::DeviceSize byteSize() const;

    /** @brief Narrows this view to tensor requirements. */
    TensorRequirementsView asTensor() const;

    /** @brief Narrows this view to storage-buffer requirements. */
    BufferRequirementsView asBuffer() const;

    /** @brief Narrows this view to image requirements. */
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

/** @brief Non-owning view of a bindable public workload resource. */
class ResourceView {
  public:
    /** @brief Returns the resource's zero-based public index. */
    uint32_t index() const;

    /** @brief Returns the resource name as a borrowed string view. */
    std::string_view name() const;

    /** @brief Returns the combined access performed by workload executables. */
    ResourceAccess access() const;

    /** @brief Returns a non-owning view of the resource requirements. */
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

/** @brief Non-owning view of a workload module. */
class ModuleView {
  public:
    /** @brief Returns the module's zero-based Workload index. */
    uint32_t index() const;

    /** @brief Returns the module name as a borrowed string view. */
    std::string_view name() const;

    /** @brief Returns the module entry-point name as a borrowed string view. */
    std::string_view entryPoint() const;

    /** @brief Returns the representation of the module code. */
    ModuleCodeKind codeKind() const;

    /** @brief Returns true if Session::bindModule() must supply this module's code. */
    bool requiresImplementation() const;

  private:
    ModuleView(const Workload *workload, uint32_t index) : workload_(workload), index_(index) {}

    friend class Workload;
    friend class ExecutableView;
    friend class PlaceholderModuleView;

    const Workload *workload_ = nullptr;
    uint32_t index_ = 0;
};

/**
 * @brief Non-owning view of a module whose code is absent from the Workload.
 *
 * Use Session::bindModule() before configuring the Session.
 */
class PlaceholderModuleView {
  public:
    /** @brief Returns the zero-based index within the placeholder-module sequence. */
    uint32_t index() const;

    /** @brief Returns the underlying module view. */
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

/**
 * @brief Non-owning view of Vulkan data graph pipeline metadata.
 *
 * This metadata is meaningful for ExecutableKind::Graph executables.
 */
class DataGraphPipelineMetadataView {
  public:
    /** @brief Returns the implementation-defined pipeline identifier. */
    std::string_view identifier() const;

    /** @brief Returns the data graph pipeline creation flags. */
    vk::PipelineCreateFlags2 flags() const;

    /** @brief Returns borrowed Vulkan specialization map entries. */
    ArrayView<const vk::SpecializationMapEntry> specializationMapEntries() const;

    /** @brief Returns the borrowed packed specialization-constant data. */
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

/** @brief Public descriptor binding metadata for one executable resource. */
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

/** @brief Non-owning view of one workload executable. */
class ExecutableView {
  public:
    /** @brief Returns the executable's zero-based Workload index. */
    uint32_t index() const;

    /** @brief Returns the executable name as a borrowed string view. */
    std::string_view name() const;

    /** @brief Returns whether this is a data graph or compute executable. */
    ExecutableKind type() const;

    /** @brief Returns the executable module. */
    ModuleView module() const;

    /** @brief Returns data graph pipeline metadata for this executable. */
    DataGraphPipelineMetadataView dataGraphPipelineMetadata() const;

    /** @brief Returns the number of descriptor bindings for public resources. */
    uint32_t interfaceDescriptorBindingCount() const;

    /** @brief Returns one public-resource descriptor binding by index. */
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

/**
 * @brief Immutable workload metadata and executable descriptions consumed by Session.
 *
 * Construct a Workload with a static factory function. Views and ranges borrow
 * its storage and must not outlive it.
 */
class Workload final {
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

    /** @brief Borrowed range over the Workload's public resources. */
    class ResourceRange {
      public:
        /** @brief Iterator that yields ResourceView values. */
        class Iterator : public RangeIteratorBase<Iterator> {
            using Base = RangeIteratorBase<Iterator>;

          public:
            /** @brief Returns the resource at the current iterator position. */
            ResourceView operator*() const;

          private:
            Iterator(const Workload *workload, uint32_t publicIndex) : Base(workload, publicIndex) {}

            friend class ResourceRange;
        };

        /** @brief Returns an iterator to the first public resource. */
        Iterator begin() const;

        /** @brief Returns the past-the-end iterator. */
        Iterator end() const;

      private:
        explicit ResourceRange(const Workload *workload) : workload_(workload) {}

        friend class Workload;

        const Workload *workload_ = nullptr;
    };

    /** @brief Borrowed range over the Workload's executables. */
    class ExecutableRange {
      public:
        /** @brief Iterator that yields ExecutableView values. */
        class Iterator : public RangeIteratorBase<Iterator> {
            using Base = RangeIteratorBase<Iterator>;

          public:
            /** @brief Returns the executable at the current iterator position. */
            ExecutableView operator*() const;

          private:
            Iterator(const Workload *workload, uint32_t index) : Base(workload, index) {}

            friend class ExecutableRange;
        };

        /** @brief Returns an iterator to the first executable. */
        Iterator begin() const;

        /** @brief Returns the past-the-end iterator. */
        Iterator end() const;

      private:
        explicit ExecutableRange(const Workload *workload) : workload_(workload) {}

        friend class Workload;

        const Workload *workload_ = nullptr;
    };

    /** @brief Borrowed range over modules whose code must be supplied by the caller. */
    class PlaceholderModuleRange {
      public:
        /** @brief Iterator that yields PlaceholderModuleView values. */
        class Iterator : public RangeIteratorBase<Iterator> {
            using Base = RangeIteratorBase<Iterator>;

          public:
            /** @brief Returns the placeholder module at the current iterator position. */
            PlaceholderModuleView operator*() const;

          private:
            Iterator(const Workload *workload, uint32_t placeholderIndex) : Base(workload, placeholderIndex) {}

            friend class PlaceholderModuleRange;
        };

        /** @brief Returns an iterator to the first placeholder module. */
        Iterator begin() const;

        /** @brief Returns the past-the-end iterator. */
        Iterator end() const;

      private:
        explicit PlaceholderModuleRange(const Workload *workload) : workload_(workload) {}

        friend class Workload;

        const Workload *workload_ = nullptr;
    };

    /***************************************************************************
     * Construction
     ***************************************************************************/

    /** @brief Loads a VGF-backed Workload from a file. */
    static Workload fromVGF(const std::filesystem::path &path);

    /**
     * @brief Decodes a VGF-backed Workload from caller-owned memory.
     *
     * The memory range `[data, data + size)` must remain valid and unchanged for
     * the lifetime of the returned Workload.
     */
    static Workload fromVGF(const void *data, std::size_t size);

    /** @brief Builds a Workload containing one standalone compute executable. */
    static Workload fromComputeShader(ComputeShaderDescription description);

    /**
     * @brief Builds a Workload containing one standalone data graph executable.
     *
     * DataGraphConstant payloads are borrowed and must remain valid for the
     * lifetime of the returned Workload.
     */
    static Workload fromDataGraph(DataGraphDescription description);

    /***************************************************************************
     * Lifetime
     ***************************************************************************/

    /** @brief Destroys the workload and invalidates all views obtained from it. */
    ~Workload();

    Workload(const Workload &) = delete;
    Workload &operator=(const Workload &) = delete;
    Workload(Workload &&) = delete;
    Workload &operator=(Workload &&) = delete;

    /***************************************************************************
     * Resource inspection
     ***************************************************************************/

    /** @brief Returns the number of bindable public resources. */
    uint32_t resourceCount() const;

    /** @brief Returns one public resource by index. */
    ResourceView resource(uint32_t publicResourceIndex) const;

    /** @brief Returns a borrowed range over all public resources. */
    ResourceRange resources() const;

    /***************************************************************************
     * Executable inspection
     ***************************************************************************/

    /** @brief Returns the number of executables in dispatch order. */
    uint32_t executableCount() const;

    /** @brief Returns one executable by index. */
    ExecutableView executable(uint32_t executableIndex) const;

    /** @brief Returns a borrowed range over executables in dispatch order. */
    ExecutableRange executables() const;

    /***************************************************************************
     * Placeholder module inspection
     ***************************************************************************/

    /** @brief Returns the number of modules that require caller-provided code. */
    uint32_t placeholderModuleCount() const;

    /** @brief Returns one placeholder module by index. */
    PlaceholderModuleView placeholderModule(uint32_t placeholderModuleIndex) const;

    /** @brief Returns a borrowed range over all placeholder modules. */
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

} // namespace mlsdk::workloadlib
