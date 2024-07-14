from pathlib import Path
import sys
from typing import List

dir_docs = Path(__file__).parent
dir_package = dir_docs.parent

print(f"{dir_package=}")
print(f"{dir_docs=}")

sys.path.insert(0, dir_package.as_posix())


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
templates_path = [(dir_docs / "_templates").as_posix()]
default_role = "code"
maximum_signature_line_length = 88


# Extensions

# autosummary
def excluded_attributes() -> List[str]:
    return [
        "model_computed_fields",
        "model_config",
        "model_fields",
        "model_extra",
        "model_fields_set",
    ]


autosummary_filename_map = {
    "redesmyn.service.Endpoint": "Endpoint.rst",
    "redesmyn.service.endpoint": "endpoint-decorator.rst",
}
autosummary_context = {
    "decorators": ["redesmyn.service.endpoint", "redesmyn.artifacts.artifact_spec"],
    "special": {"redesmyn.schema.Schema": ["DataFrame"]},
    "show_inheritance": [
        "redesmyn.artifacts.ArtifactSpec",
        "redesmyn.artifacts.LatestKey",
        "redesmyn.artifacts.CacheConfig",
        "redesmyn.artifacts.Cron",
        "redesmyn.artifacts.PathTemplate",
    ],
    "no_inherited_members": {
        "redesmyn.artifacts.ArtifactSpec": ["BaseModel"],
        "redesmyn.artifacts.LatestKey": ["BaseModel"],
        "redesmyn.artifacts.CacheConfig": ["BaseModel"],
        "redesmyn.artifacts.Cron": ["BaseModel"],
        "redesmyn.artifacts.PathTemplate": ["Path"],
    },
    "exclude_attributes": {
        "redesmyn.artifacts.ArtifactSpec": excluded_attributes(),
        "redesmyn.artifacts.LatestKey": excluded_attributes(),
        "redesmyn.artifacts.CacheConfig": excluded_attributes(),
        "redesmyn.artifacts.Cron": excluded_attributes(),
        "redesmyn.artifacts.PathTemplate": [
            "braceidpattern",
            "delimiter",
            "flags",
            "idpattern",
            "pattern",
        ]
    },
}


# HTML

html_static_path = ["_static"]
html_css_files = ["style.css"]
html_show_sourcelink = False
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 4,
}
