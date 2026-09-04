#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# ML Workload Library for Vulkan project config
MLWORKLOADLIB_project = "ML Workload Library for Vulkan"
copyright = "2026, Arm Limited and/or its affiliates <open-source-office@arm.com>"
author = "Arm Limited"
git_repo_tool_url = "https://gerrit.googlesource.com/git-repo"

# Set home project name
project = MLWORKLOADLIB_project

rst_epilog = """
.. |MLWORKLOADLIB_project| replace:: %s
.. |git_repo_tool_url| replace:: %s
""" % (
    MLWORKLOADLIB_project,
    git_repo_tool_url,
)

# Enabled extensions
extensions = [
    "breathe",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

# Disable converting double-dash to typographical en-dash
smartquotes_action = "qe"

# Disable superfluous warnings
suppress_warnings = [
    "autosectionlabel.*",
    "myst.xref_missing",
    "myst.header",
]

# Breathe Configuration
breathe_projects = {"WorkloadLib": "../generated/xml"}
breathe_default_project = "Workload-Lib"
breathe_domain_by_extension = {"h": "c"}

# Enable RTD theme
html_theme = "sphinx_rtd_theme"

tags.add("WITH_BASE_MD")
