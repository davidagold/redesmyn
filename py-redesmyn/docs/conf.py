from pathlib import Path
import sys


sys.path.insert(0, Path("../..").resolve().as_posix())


# Project

project = "Redesmyn"
author = "David A. Gold"
copyright = f"2023, {author}"


# General

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
]
templates_path = ["_templates"]
default_role = "code"
maximum_signature_line_length = 88


# HTML

html_static_path = ["_static"]
html_css_files = ["style.css"]
html_show_sourcelink = False
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 4,
}
