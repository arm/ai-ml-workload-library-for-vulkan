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
- Python 3.11 or later and `uv` 0.9.26. Python tooling dependencies are
  declared in `pyproject.toml` and locked in `uv.lock`.
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

Build the HTML documentation independently of the native library. This requires
CMake 3.25 or later, Ninja, Doxygen 1.9.1 or later, and the locked Python tooling
dependencies (including Sphinx, Breathe, MyST, and the Read the Docs theme).
No C/C++ compiler or native dependencies are required.

Run these commands from the component root:

```sh
uv sync --locked --no-install-project --no-default-groups --group tooling
uv run --no-sync cmake -S docs -B build-docs -G Ninja
cmake --build build-docs
```

The generated documentation is available at `build-docs/docs/out/index.html`.
Choose another documentation build directory with `-B` and use the same directory
in the build command. Documentation inputs are staged under that directory's
`docs/in`, `docs/generated`, and `docs/sources`; Doxygen XML is written to
`docs/doxygen/xml`.

During migration, `scripts/build.py --doc` continues to build both the native
library and documentation using the existing path, with HTML at
`build/docs/out/index.html`. The native CMake option
`ML_WORKLOAD_LIB_BUILD_DOCS=ON` and the `mlworkloadlib_doc` target remain available.
Use separate build directories for native and standalone documentation projects;
their CMake caches cannot be shared.

## Code coverage

Sync the locked Python tooling dependencies, then run the unit tests on Linux
with GCC coverage instrumentation:

```sh
uv sync --locked --no-install-project --no-default-groups --group test --group tooling
uv run --no-sync python scripts/build.py --coverage --build-dir build-coverage
```

The coverage command writes a detailed HTML report to
`build-coverage/coverage/index.html` and a machine-readable JSON summary to
`build-coverage/coverage/summary.json`. Coverage is collected for the library
sources and public headers; tests and bundled dependencies are excluded from
the report.

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
