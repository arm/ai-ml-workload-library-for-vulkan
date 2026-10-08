/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlworkloadlib/binding_set.hpp"
#include "mlworkloadlib/compiled_execution.hpp"
#include "mlworkloadlib/prepared_execution.hpp"
#include "mlworkloadlib/workload.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlsdk::workloadlib {

/** @brief Non-owning information about one graph-session memory allocation. */
struct DataGraphSessionMemoryInfo {
    vk::DeviceMemory memory = nullptr;
    vk::DeviceSize size = 0;
    vk::DataGraphPipelineSessionBindPointARM bindPoint{};
    uint32_t objectIndex = 0;
    vk::MemoryPropertyFlags memoryProperties;
    bool neuralStatistics = false;
};

/** @brief Raw data returned for one graph-pipeline property. */
struct DataGraphPipelinePropertyData {
    vk::DataGraphPipelinePropertyARM property{};
    bool isText = false;
    std::vector<uint8_t> data;
};

/*******************************************************************************
 * Session
 *******************************************************************************/

/**
 * @brief Configured runtime state for a Workload on a Context.
 *
 * Context and Workload are borrowed and must outlive the Session.
 */
class Session final {
  public:
    /***************************************************************************
     * Creation and lifetime
     **************************************************************************/

    /** @brief Creates an unconfigured Session with its own compiled state. */
    Session(Context &context, const Workload &workload, SessionOptions options = {});

    /** @brief Creates an unconfigured Session sharing state with @p compiledExecution. */
    Session(Context &context, const Workload &workload, CompiledExecution &compiledExecution);

    /** @brief Releases session-specific objects and its reference to compiled state. */
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    /***************************************************************************
     * Configuration
     **************************************************************************/

    /** @brief Supplies or replaces code for a placeholder module before configure(). */
    void bindModule(PlaceholderModuleView placeholderModule, ModuleImplementation implementation);

    /** @brief Builds or reuses pipelines and layouts, then creates per-Session data graph and command objects. */
    void configure();

    /***************************************************************************
     * Graph diagnostics
     **************************************************************************/

    /** @brief Returns the properties advertised by one configured graph pipeline. */
    std::vector<vk::DataGraphPipelinePropertyARM> dataGraphPipelineProperties(uint32_t executableIndex) const;

    /** @brief Retrieves raw data for one advertised graph-pipeline property. */
    DataGraphPipelinePropertyData dataGraphPipelineProperty(uint32_t executableIndex,
                                                            vk::DataGraphPipelinePropertyARM property) const;

    /** @brief Returns the number of owned memory allocations for one graph session. */
    uint32_t dataGraphSessionMemoryCount(uint32_t executableIndex) const;

    /** @brief Returns a non-owning view of one graph-session memory allocation. */
    DataGraphSessionMemoryInfo dataGraphSessionMemory(uint32_t executableIndex, uint32_t memoryIndex) const;

    /***************************************************************************
     * Factories
     **************************************************************************/

    /** @brief Creates an empty BindingSet associated with this Session. */
    BindingSet createBindingSet();

    /** @brief Validates and snapshots bindings for execution. */
    PreparedExecution prepare(const BindingSet &bindings);

  private:
    // Implementation type
    struct Impl;

    // Friends
    friend class BindingSet;
    friend class PreparedExecution;

    // Implementation access
    Impl &sessionImpl() noexcept;
    const Impl &sessionImpl() const noexcept;

    // Stored state
    std::unique_ptr<Impl> impl_;
};

} // namespace mlsdk::workloadlib
