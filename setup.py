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
from setuptools.dist import Distribution

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

        build_args = [
            "--build-dir",
            str(native_build_dir),
            "--install",
            str(native_install_dir),
            "--build-shared",
        ]
        build_args.extend(["--package-version", self.distribution.get_version()])

        result = build_workload_lib(build_args)
        if result:
            raise RuntimeError(f"Workload Lib native build failed with code {result}")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return ("py3", "none", platform_tag)


setup(
    cmdclass={"build": Build, "build_py": BuildPy, "bdist_wheel": BDistWheel},
    distclass=BinaryDistribution,
)
