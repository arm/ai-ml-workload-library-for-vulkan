#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path


def get_cmake_prefix() -> Path:
    """Return the prefix containing the packaged headers, libraries, and CMake files."""
    return Path(__file__).resolve().parent / "binaries"


__all__ = ["get_cmake_prefix"]
