# ML Workload Library for Vulkan® — Release Notes

## Unreleased – *Initial Public Release*

## Purpose

The ML Workload Library for Vulkan® provides an embeddable C++17 API for constructing,
inspecting, and executing Vulkan® ML workloads. Applications can retain ownership
of their Vulkan® objects and bound resources, or use library-owned Vulkan® contexts
and resource allocations, while the library manages the workload's execution
state.

## Features

### Workloads and Modules

- Load VGF workloads from files or caller-owned memory, including composed
  compute and Vulkan® data-graph execution sequences.
- Construct standalone compute-shader and Vulkan® data-graph workloads.
- Use embedded SPIR-V™ modules, optionally compile GLSL or HLSL source modules,
  and supply implementations for placeholder modules before configuration.
- Inspect workload resources, executables, modules, bindings, dispatches, and
  data-graph metadata through non-owning C++ views.

### Resource Binding and Execution

- Wrap caller-owned Vulkan® RAII instance, physical device, device, queue, and
  queue-family state in a non-owning context, or create a runtime-owned context
  with additional device extension and feature requirements.
- Allocate workload-compatible tensors, storage buffers, images, and backing
  memory as runtime-owned RAII objects.
- Configure reusable sessions and bind caller-owned or runtime-owned tensors,
  storage buffers, images, and their memory where required.
- Optionally provide an image's current layout when binding it so prepared
  execution can transition it to the layout required by the workload.
- Prepare immutable binding snapshots, then either run them with library-managed
  command state or record them into a caller-owned command buffer.
- Validate configuration order, workload compatibility, required bindings,
  module code, resource metadata, and dispatch dimensions.

### Build and Integration

- Optionally build the ML Workload Library for Vulkan® as a shared library with
  `ML_WORKLOAD_LIB_BUILD_SHARED` for ABI and API compatibility analysis.
- Build C++ samples for inspecting and running VGF workloads, constructing a
  standalone data-graph workload, running a standalone GLSL compute workload,
  binding placeholder modules, and wrapping application-owned Vulkan® context
  objects.

## Supported Platforms

The following platform combinations are supported:

- Linux - AArch64 and x86-64
- Windows® - x86-64
- Darwin - AArch64 (experimental)
- Android™ - AArch64 (experimental)
