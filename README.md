# ML Workload Library for Vulkan®

The Workload Library provides an **experimental** C++ API for constructing executable
ML workloads for Vulkan® and running them through a common resource-binding
and execution model.

## Public API

Include the installed public headers that your application requires:

```cpp
#include <mlworkloadlib/workload.hpp>
#include <mlworkloadlib/context.hpp>
#include <mlworkloadlib/session.hpp>
```

The API exposes:

- `mlsdk::workloadlib::Workload`
- `mlsdk::workloadlib::Context`
- `mlsdk::workloadlib::Session`
- `mlsdk::workloadlib::BindingSet`
- `mlsdk::workloadlib::PreparedExecution`

You can create the following workloads:

- VGF-backed workloads with `Workload::fromVGF(...)`
- Standalone compute shader workloads with `Workload::fromComputeShader(...)`
- Standalone Vulkan® data graph workloads with `Workload::fromDataGraph(...)`

Use `Context::wrap(...)` to provide application-owned Vulkan® objects, or
use `Context::create()` to create a runtime-owned Vulkan® context.

## Building the ML Workload Library for Vulkan® from source

The build system must have:

- C/C++ 17 compiler: GCC or Clang on Linux, Clang on Darwin, or MSVC on
  Windows®.
- CMake 3.25 or later.
- Ninja 1.8.2 or later.
- Python 3.10 or later. Required python libraries for building are listed in
  `tooling-requirements.txt`.
- Doxygen 1.9.1 or later. (When building documentation)

To create an archive containing the build artifacts, pass the `--package-type`
option with an archive type such as `zip` or `tgz`. Use `--package-dir` to
choose where the archive is written; by default, packages are written to the
build directory.

For more command line options, see the help output:

```bash
python3 scripts/build.py --help
```

## Build-tree usage

Enable the library when configuring the standalone project:

```sh
cmake -S . -B build
```

Link one of the build-tree targets:

```cmake
target_link_libraries(my_target PRIVATE mlworkloadlib)
```

Alternatively, link a namespaced target in the same build:

```cmake
target_link_libraries(my_target PRIVATE MLWorkloadLibraryForVulkan::mlworkloadlib)
```

## Installed package usage

When the ML Workload Library for Vulkan® is installed, the package exports `mlworkloadlib` through the
`MLWorkloadLibraryForVulkan` package. Downstream consumers must use
`MLWorkloadLibraryForVulkan::mlworkloadlib`.

```cmake
find_package(MLWorkloadLibraryForVulkan CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE MLWorkloadLibraryForVulkan::mlworkloadlib)
```

The package target requires the `VGF` and `VulkanHeaders` packages to be
discoverable at configure time.

## Optional VGF support

VGF workload support is enabled by default. Disable it when the library is only
needed for programmatically constructed workloads:

```sh
python3 scripts/build.py --disable-vgf-support
```

For direct CMake configuration, set `ML_WORKLOAD_LIB_ENABLE_VGF_SUPPORT=OFF`.
Such builds do not locate or link VGF Lib. Calls to `Workload::fromVGF(...)`
throw an exception indicating that VGF support is unavailable.

## Optional source module support

The base library supports SPIR-V™ modules. You can choose to enable GLSL and
HLSL source compilation when you build the library:

```sh
python3 scripts/build.py --enable-glsl-support
python3 scripts/build.py --enable-hlsl-support
```

Use `--glslang-path` or `--dxc-path` when the dependency is not in the default
location. For direct CMake configuration, use
`ML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT` with `GLSLANG_PATH`, or
`ML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT` with `DXC_PATH`.

Applications can query the source backends available in the linked library with
`mlsdk::workloadlib::supports(...)`.

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

Khronos® and Vulkan® are registered trademarks, and SPIR-V™ is a trademark of
[The Khronos Group Inc.](https://www.khronos.org/legal/trademarks/).
