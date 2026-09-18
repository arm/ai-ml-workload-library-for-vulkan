#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import os
import pathlib
import platform
import re
import subprocess
import sys
from datetime import datetime
from datetime import timezone

try:
    import argcomplete
except ImportError:
    argcomplete = None

ML_WORKLOAD_LIB_DIR = pathlib.Path(__file__).parent / ".."
ML_WORKLOAD_LIB_DIR = ML_WORKLOAD_LIB_DIR.resolve()
DEPENDENCY_DIR = ML_WORKLOAD_LIB_DIR / ".." / ".." / "dependencies"
DEPENDENCY_DIR = DEPENDENCY_DIR.resolve()
CMAKE_TOOLCHAIN_PATH = ML_WORKLOAD_LIB_DIR / "cmake" / "toolchain"


def absolute(path):
    return pathlib.Path(path).resolve().as_posix()


class Builder:
    """
    A class that builds the ML Workload Library for Vulkan.

    Parameters
    ----------
    args : 'dict'
        Dictionary with arguments to build the ML Workload Library for Vulkan.
    """

    def __init__(self, args) -> None:
        self.build_dir = str(pathlib.Path(args.build_dir).resolve())
        self.prefix_path = args.prefix_path
        self.threads = args.threads
        self.run_tests = args.test
        self.lint = args.lint
        self.build_type = args.build_type
        self.doc_only = args.doc_only
        self.doc = args.doc or self.doc_only
        self.target_platform = args.target_platform
        self.fuzzer = args.fuzzer
        if (
            self.fuzzer
            and self.target_platform == "host"
            and platform.system() == "Linux"
        ):
            self.target_platform = "linux-clang"

        self.cmake_toolchain_for_android = args.cmake_toolchain_for_android
        self.vulkan_headers_path = absolute(args.vulkan_headers_path)
        self.vgf_lib_path = absolute(args.vgf_lib_path)
        self.flatbuffers_path = absolute(args.flatbuffers_path)
        self.glslang_path = absolute(args.glslang_path)
        self.dxc_path = absolute(args.dxc_path)
        self.spirv_tools_path = absolute(args.spirv_tools_path)
        self.spirv_headers_path = absolute(args.spirv_headers_path)
        self.gtest_path = absolute(args.gtest_path)
        self.enable_glsl_support = args.enable_glsl_support
        self.enable_hlsl_support = args.enable_hlsl_support
        self.disable_vgf_support = args.disable_vgf_support
        self.install = args.install

        self.package_dir = args.package_dir or self.build_dir
        self.package_tgz = "tgz" in args.package_type
        self.package_zip = "zip" in args.package_type
        self.package_pip = "pip" in args.package_type
        self.package_release_pip = "release-pip" in args.package_type
        self.package_version = args.package_version
        self.package_source_tgz = "source-tgz" in args.package_type
        self.package_source_zip = "source-zip" in args.package_type

        if self.package_release_pip:
            self.package_pip = True

        self.pip_install = str(
            ML_WORKLOAD_LIB_DIR / "pip_package" / "mlworkloadlib" / "binaries"
        )
        if not self.install and self.package_pip:
            self.install = self.pip_install

    def setup_platform_build(self, cmake_cmd):
        system = platform.system()
        if self.target_platform == "host":
            if system == "Linux":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'gcc.cmake'}"
                )
                return True

            if system == "Darwin":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'clang.cmake'}"
                )
                return True

            if system == "Windows":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'windows-msvc.cmake'}"
                )
                cmake_cmd.append("-DMSVC=ON")
                return True

            print(f"ERROR: Unsupported host platform {system}", file=sys.stderr)
            return False

        if self.target_platform == "linux-clang":
            if system != "Linux":
                print(
                    f"ERROR: target {self.target_platform} only supported on Linux. Host platform {system}",
                    file=sys.stderr,
                )
                return False
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'clang.cmake'}"
            )
            return True

        if self.target_platform == "aarch64":
            if system != "Linux":
                print(
                    f"ERROR: target {self.target_platform} only supported on Linux. Host platform {system}",
                    file=sys.stderr,
                )
                return False
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'linux-aarch64-gcc.cmake'}"
            )
            cmake_cmd.append("-DHAVE_CLONEFILE=0")
            cmake_cmd.append("-DBUILD_TOOLS=OFF")
            cmake_cmd.append("-DBUILD_REGRESS=OFF")
            cmake_cmd.append("-DBUILD_EXAMPLES=OFF")
            cmake_cmd.append("-DBUILD_DOC=OFF")

            cmake_cmd.append("-DBUILD_WSI_WAYLAND_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XLIB_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XCB_SUPPORT=OFF")
            return True

        if self.target_platform == "android":
            if system != "Linux":
                print(
                    f"ERROR: target {self.target_platform} only supported on Linux. Host platform {system}",
                    file=sys.stderr,
                )
                return False
            print(
                "WARNING: Cross-compiling ML Workload Library for Android is currently an experimental feature."
            )
            if not self.cmake_toolchain_for_android:
                print(
                    "ERROR: No toolchain path specified for Android cross-compilation",
                    file=sys.stderr,
                )
                return False

            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={self.cmake_toolchain_for_android}"
            )
            cmake_cmd.append("-DANDROID_ABI=arm64-v8a")
            cmake_cmd.append("-DANDROID_PLATFORM=android-21")
            cmake_cmd.append("-DANDROID_ALLOW_UNDEFINED_SYMBOLS=ON")
            return True

        print(
            f"ERROR: Incorrect target platform option: {self.target_platform}",
            file=sys.stderr,
        )
        return False

    def generate_cmake_package(self, generator, source_package=False):
        config_file = (
            "CPackSourceConfig.cmake" if source_package else "CPackConfig.cmake"
        )

        cmake_package_cmd = [
            "cpack",
            "--config",
            f"{self.build_dir}/{config_file}",
            "-C",
            self.build_type,
            "-G",
            generator,
            "-B",
            self.package_dir,
            "-D",
            "CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF",
        ]
        subprocess.run(cmake_package_cmd, check=True)

    def run(self):
        cmake_setup_cmd = [
            "cmake",
            "-S",
            str(ML_WORKLOAD_LIB_DIR),
            "-B",
            self.build_dir,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'gcc.cmake'}",
            f"-DVULKAN_HEADERS_PATH={self.vulkan_headers_path}",
            "-G",
            "Ninja",
        ]

        if self.disable_vgf_support:
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_ENABLE_VGF_SUPPORT=OFF")
        else:
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_ENABLE_VGF_SUPPORT=ON")
            cmake_setup_cmd.append(f"-DML_SDK_VGF_LIB_PATH={self.vgf_lib_path}")
            cmake_setup_cmd.append(f"-DFLATBUFFERS_PATH={self.flatbuffers_path}")

        if self.fuzzer:
            if self.target_platform in ["android", "aarch64"]:
                print(
                    "ERROR: fuzzer builds require a host Clang/libFuzzer toolchain",
                    file=sys.stderr,
                )
                return 1
            if platform.system() == "Windows":
                print(
                    "ERROR: fuzzer builds are not supported on Windows", file=sys.stderr
                )
                return 1
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_ENABLE_FUZZER=ON")

        if self.prefix_path:
            cmake_setup_cmd.append(f"-DCMAKE_PREFIX_PATH={self.prefix_path}")

        if self.run_tests:
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_BUILD_TESTS=ON")
            cmake_setup_cmd.append(f"-DSPIRV_TOOLS_PATH={self.spirv_tools_path}")
            cmake_setup_cmd.append(
                f"-DSPIRV-Headers_SOURCE_DIR={self.spirv_headers_path}"
            )
            cmake_setup_cmd.append(f"-DGTEST_PATH={self.gtest_path}")

        if self.lint:
            cmake_setup_cmd.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

        if self.doc:
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_BUILD_DOCS=ON")

        if self.enable_glsl_support:
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_ENABLE_GLSL_SUPPORT=ON")
            cmake_setup_cmd.append(f"-DGLSLANG_PATH={self.glslang_path}")

        if self.enable_hlsl_support:
            cmake_setup_cmd.append("-DML_WORKLOAD_LIB_ENABLE_HLSL_SUPPORT=ON")
            cmake_setup_cmd.append(f"-DDXC_PATH={self.dxc_path}")

        if self.package_version:
            cmake_setup_cmd.append(f"-DML_SDK_PACKAGE_VERSION={self.package_version}")

        if not self.setup_platform_build(cmake_setup_cmd):
            return 1

        cmake_build_cmd = [
            "cmake",
            "--build",
            self.build_dir,
            "-j",
            str(self.threads),
            "--config",
            self.build_type,
        ]
        if self.doc_only:
            cmake_build_cmd.extend(["--target", "mlworkloadlib_doc"])

        if self.fuzzer and not self.run_tests:
            cmake_build_cmd.extend(["--target", "mlworkloadlib_fuzzer"])

        try:
            subprocess.run(cmake_setup_cmd, check=True)
            subprocess.run(cmake_build_cmd, check=True)

            if self.lint:
                src_dirs = [
                    f"{ML_WORKLOAD_LIB_DIR / 'src'}",
                    f"{ML_WORKLOAD_LIB_DIR / 'utils'}",
                    f"{ML_WORKLOAD_LIB_DIR / 'samples'}",
                    f"{ML_WORKLOAD_LIB_DIR / 'fuzz'}",
                    f"{ML_WORKLOAD_LIB_DIR / 'tests'}",
                ]
                pathlib.Path(self.build_dir, "cppcheck").mkdir(
                    parents=True, exist_ok=True
                )

                lint_cmd = [
                    "cppcheck",
                    f"-j{str(self.threads)}",
                    "--std=c++17",
                    "--error-exitcode=1",
                    "--inline-suppr",
                    f"--cppcheck-build-dir={self.build_dir}/cppcheck",
                    "--enable=information,performance,portability,style",
                    "--suppress=missingIncludeSystem",
                    "--suppress=missingInclude",
                    "--suppress=noValidConfiguration",
                    "--suppress=unknownMacro",
                    "--suppress=unmatchedSuppression",
                    "--suppress=useStlAlgorithm",
                    "--suppress=*:MachineIndependent*",
                    f"--suppress=*:{DEPENDENCY_DIR}*",
                ] + src_dirs
                subprocess.run(lint_cmd, check=True)

                clang_tidy_cmd = [
                    "run-clang-tidy",
                    "-quiet",
                    f"-j{self.threads}",
                    f"-p{self.build_dir}",
                    "-header-filter=workload-lib/.*",
                ] + src_dirs
                subprocess.run(clang_tidy_cmd, check=True)

            if self.install:
                cmake_install_cmd = [
                    "cmake",
                    "--install",
                    self.build_dir,
                    "--prefix",
                    self.install,
                    "--config",
                    self.build_type,
                ]
                subprocess.run(cmake_install_cmd, check=True)

            if self.run_tests:
                test_cmd = [
                    "ctest",
                    "--test-dir",
                    self.build_dir,
                    "-j",
                    str(self.threads),
                    "--output-on-failure",
                ]
                subprocess.run(test_cmd, check=True)

            if self.package_tgz:
                self.generate_cmake_package("TGZ")

            if self.package_zip:
                self.generate_cmake_package("ZIP")

            if self.package_source_tgz:
                self.generate_cmake_package("TGZ", True)

            if self.package_source_zip:
                self.generate_cmake_package("ZIP", True)

            if self.package_pip:
                if self.install != self.pip_install:
                    subprocess.run(
                        [
                            "cmake",
                            "--install",
                            self.build_dir,
                            "--prefix",
                            self.pip_install,
                            "--config",
                            self.build_type,
                        ],
                        check=True,
                    )

                package_version = ""
                if self.package_version:
                    package_version = self.package_version
                else:
                    package_version = (
                        "" if self.package_release_pip else get_package_version()
                    )

                build_env = os.environ.copy()
                build_env[
                    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AI_ML_WORKLOAD_LIBRARY_FOR_VULKAN"
                ] = package_version
                if package_version:
                    build_env["ML_SDK_PACKAGE_VERSION"] = package_version
                build_env["ML_WORKLOAD_LIB_SKIP_NATIVE_BUILD"] = "1"

                result = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "build",
                        "--outdir",
                        str(ML_WORKLOAD_LIB_DIR / "pip_package" / "dist"),
                        str(ML_WORKLOAD_LIB_DIR),
                    ],
                    env=build_env,
                    cwd=ML_WORKLOAD_LIB_DIR,
                )
                result.communicate()
                if result.returncode != 0:
                    print("ERROR: Failed to generate pip package", file=sys.stderr)
                    return 1

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: Build failed with error: {e}", file=sys.stderr)
            return 1

        return 0


def get_package_version():
    pyproject = (ML_WORKLOAD_LIB_DIR / "pyproject.toml").read_text()

    regex_result = re.search(r'fallback_version\s*=\s*"([^"]+)"', pyproject)
    if not regex_result:
        raise RuntimeError("fallback_version not found")

    base_version = regex_result.group(1)

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    return f"{base_version}.dev{date_tag}"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        help="Name of folder where to build the library. Default: %(default)s",
        default=f"{ML_WORKLOAD_LIB_DIR / 'build'}",
    )
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        help="Number of threads to use for building. Default: %(default)s",
        default=16,
    )
    parser.add_argument(
        "--prefix-path",
        help="Path to prefix directory.",
    )
    parser.add_argument(
        "-t",
        "--test",
        help="Run unit tests after build. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--lint",
        help="Run linter. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fuzzer",
        help="Build the Clang/libFuzzer target. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--build-type",
        help="Type of build to perform. Default: %(default)s",
        default="Release",
    )
    doc_group = parser.add_mutually_exclusive_group()
    doc_group.add_argument(
        "--doc",
        help="Build documentation. Default: %(default)s",
        action="store_true",
        default=False,
    )
    doc_group.add_argument(
        "--doc-only",
        help="Only build documentation. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--target-platform",
        help="Specify the target build platform. Default: %(default)s",
        choices=["host", "android", "aarch64", "linux-clang"],
        default="host",
    )
    parser.add_argument(
        "--cmake-toolchain-for-android",
        help="Path to the cmake compiler toolchain. Default: %(default)s",
        default="",
    )
    parser.add_argument(
        "--vulkan-headers-path",
        help="Path to the vulkan headers folder. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'Vulkan-Headers'}",
    )
    parser.add_argument(
        "--vgf-lib-path",
        help="Path to the ai-ml-sdk-vgf-library repo. Default: %(default)s",
        default=f"{ML_WORKLOAD_LIB_DIR / '..' / 'vgf-lib'}",
    )
    parser.add_argument(
        "--disable-vgf-support",
        help="Disable VGF workload support",
        action="store_true",
    )
    parser.add_argument(
        "--flatbuffers-path",
        help="Path to flatbuffers repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'flatbuffers'}",
    )
    parser.add_argument(
        "--glslang-path",
        help="Path to glslang repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'glslang'}",
    )
    parser.add_argument(
        "--dxc-path",
        help="Path to the dxc (DirectXShaderCompiler) repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'DirectXShaderCompiler'}",
    )
    parser.add_argument(
        "--enable-glsl-support",
        help="Enable GLSL source module support",
        action="store_true",
    )
    parser.add_argument(
        "--enable-hlsl-support",
        help="Enable HLSL source module support",
        action="store_true",
    )
    parser.add_argument(
        "--spirv-tools-path",
        help="Path to spirv-tools repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'SPIRV-Tools'}",
    )
    parser.add_argument(
        "--spirv-headers-path",
        help="Path to spirv-headers repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'SPIRV-Headers'}",
    )
    parser.add_argument(
        "--gtest-path",
        help="Path to googletest repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'googletest'}",
    )
    parser.add_argument(
        "--install",
        help="Install build artifacts into a provided location",
    )
    parser.add_argument(
        "--package-dir",
        help=(
            "Specify location for packages to be created. Defaults to the build "
            "directory."
        ),
        default="",
    )
    parser.add_argument(
        "--package-type",
        choices=["zip", "tgz", "pip", "source-zip", "source-tgz", "release-pip"],
        action="append",
        help="Create a package of a certain type",
        default=[],
    )
    parser.add_argument(
        "--package-version",
        help="Manually specify package version number",
        default="",
    )

    if argcomplete and argv is None:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def build(argv=None):
    return Builder(parse_arguments(argv)).run()


def main():
    sys.exit(build())


if __name__ == "__main__":
    main()
