# ML Workload Library for Vulkan®

`mlworkloadlib` provides a public C++ API for constructing executable ML workloads
for Vulkan® and running them through a common resource-binding and execution
model.

## Public API

The installed public headers are included directly as needed:

```cpp
#include <mlworkloadlib/workload.hpp>
#include <mlworkloadlib/context.hpp>
#include <mlworkloadlib/session.hpp>
```

The API exposes `mlworkloadlib::Workload`, `mlworkloadlib::Context`,
`mlworkloadlib::Session`, `mlworkloadlib::BindingSet`, and
`mlworkloadlib::PreparedExecution`. Supported workloads are VGF-backed workloads
created with `Workload::fromVGF(...)` and standalone compute shader workloads
created with `Workload::fromComputeShader(...)`, and standalone Vulkan® data
graph workloads created with `Workload::fromDataGraph(...)`; callers provide
application-owned Vulkan® objects through `Context::wrap(...)`, or let
`Context::create()` create a runtime-owned Vulkan® context.

## Build-tree usage

Enable the library when configuring the standalone project:

```sh
cmake -S . -B build
```

Link one of the build-tree targets:

```cmake
target_link_libraries(my_target PRIVATE mlworkloadlib)
```

or, if you prefer a namespaced target in the same build:

```cmake
target_link_libraries(my_target PRIVATE MLWorkloadLibraryForVulkan::mlworkloadlib)
```

## Installed package usage

When the ML Workload Library for Vulkan® is installed, the package exports `mlworkloadlib` through the
`MLWorkloadLibraryForVulkan` package. Downstream consumers should use
`MLWorkloadLibraryForVulkan::mlworkloadlib`.

```cmake
find_package(MLWorkloadLibraryForVulkan CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE MLWorkloadLibraryForVulkan::mlworkloadlib)
```

The package target expects the `VGF` and `VulkanHeaders` packages to be discoverable at configure time.

## Optional source module support

SPIR-V™ modules are supported by the base library. GLSL and HLSL source
compilation are optional and can be enabled when building:

```sh
python3 scripts/build.py --enable-glsl-support
python3 scripts/build.py --enable-hlsl-support
```

Use `--glslang-path` or `--dxc-path` when the dependency is not in the default
location. For direct CMake configuration, use
`ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT` with `GLSLANG_PATH`, or
`ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT` with `DXC_PATH`.

Applications can query the source backends available in the linked library with
`mlworkloadlib::supports(...)`.

## Documentation

Install Doxygen and the Python documentation tool dependencies and build the HTML documentation:

```sh
python3 -m pip install -r tooling-requirements.txt
python3 scripts/build.py --doc
```

The generated documentation is available at `build/docs/out/index.html`.

## Consumption model

- `mlworkloadlib` is install-tree consumable today via `find_package(MLWorkloadLibraryForVulkan)`.
- `mlworkloadlib` depends on external packages (`VGF`, `VulkanHeaders`), and those
  dependencies are resolved via the standalone package config.

## License

The ML Workload Library for Vulkan® is distributed under the software licenses in
the `LICENSES` directory.

## Trademark notice

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US
and/or elsewhere.

Khronos®, Vulkan®, and SPIR-V™ are trademarks of the
[Khronos® Group](https://www.khronos.org/legal/trademarks).
