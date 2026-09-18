#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import platform
import shutil
import sys

from setuptools import setup
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


ML_WORKLOAD_LIB_DIR = pathlib.Path(__file__).resolve().parent
SKIP_NATIVE_BUILD_ENV = "ML_WORKLOAD_LIB_SKIP_NATIVE_BUILD"


def packaged_libraries():
    binaries_dir = ML_WORKLOAD_LIB_DIR / "pip_package" / "mlworkloadlib" / "binaries"
    system = platform.system()
    if system == "Windows":
        return [
            binaries_dir / "lib" / "mlworkloadlib_static.lib",
            binaries_dir / "lib" / "mlworkloadlib.lib",
            binaries_dir / "bin" / "mlworkloadlib.dll",
        ]
    if system == "Darwin":
        return [
            binaries_dir / "lib" / "libmlworkloadlib.a",
            binaries_dir / "lib" / "libmlworkloadlib.dylib",
        ]
    return [
        binaries_dir / "lib" / "libmlworkloadlib.a",
        binaries_dir / "lib" / "libmlworkloadlib.so",
    ]


class Build(setuptools_build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(pathlib.Path("build") / "python")


class BuildPy(build_py):
    def run(self):
        super().run()

        if os.environ.get(SKIP_NATIVE_BUILD_ENV) == "1" or all(
            library.is_file() for library in packaged_libraries()
        ):
            return

        dependency_dir = ML_WORKLOAD_LIB_DIR.parent.parent / "dependencies"
        if not dependency_dir.is_dir():
            raise RuntimeError(
                "The Workload Lib native build requires an ML SDK checkout. "
                f"Missing: {dependency_dir}"
            )

        missing_tools = [tool for tool in ("cmake", "ninja") if not shutil.which(tool)]
        if missing_tools:
            raise RuntimeError(
                "The Workload Lib native build requires: " + ", ".join(missing_tools)
            )

        sys.path.insert(0, str(ML_WORKLOAD_LIB_DIR))
        from scripts.build import build as build_workload_lib

        build_command = self.get_finalized_command("build")
        native_build_dir = pathlib.Path(build_command.build_temp) / "workload_lib"
        native_install_dir = pathlib.Path(self.build_lib) / "mlworkloadlib" / "binaries"

        result = build_workload_lib(
            [
                "--build-dir",
                str(native_build_dir),
                "--install",
                str(native_install_dir),
                "--build-shared",
            ]
        )
        if result:
            raise RuntimeError(f"Workload Lib native build failed with code {result}")


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        system = platform.system()
        machine = platform.machine()
        if system == "Windows":
            assert machine == "AMD64"
            platform_name = "win_amd64"
        elif system == "Linux":
            if machine == "aarch64":
                platform_name = "manylinux2014_aarch64"
            else:
                assert machine == "x86_64"
                platform_name = "manylinux2014_x86_64"
        elif system == "Darwin":
            assert machine == "arm64"
            platform_name = "macosx_11_0_arm64"
        else:
            raise RuntimeError(f"Unsupported platform: {system} {machine}")

        return ("py3", "none", platform_name)


setup(cmdclass={"build": Build, "build_py": BuildPy, "bdist_wheel": BDistWheel})
