import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import toml


def convert_to_pep508(name, version):
    if version.startswith("^"):
        version = f">={version[1:]},<{bump_major_version(version[1:])}"
    elif version.startswith("~"):
        version = f">={version[1:]},<{bump_minor_version(version[1:])}"
    elif version.startswith(">") or version.startswith("<"):
        pass
    else:
        version = f"=={version}"
    return f"{name}{version}"


def bump_major_version(version):
    major, *rest = version.split(".")
    return f"{int(major) + 1}"


def bump_minor_version(version):
    parts = version.split(".")
    if len(parts) == 1:
        return f"{int(parts[0]) + 1}"
    major, minor, *rest = parts
    return f"{major}.{int(minor) + 1}"


def parse_poetry_dependencies(pyproject: Dict) -> List[str]:
    poetry_dependencies = pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
    pep508_dependencies = []
    for name, version in poetry_dependencies.items():
        if name == "python":
            continue  # Skip python version specification
        pep508_dependencies.append(convert_to_pep508(name, version))

    return pep508_dependencies


if __name__ == "__main__":
    pyproject_file = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_file, "r") as f:
        pyproject = defaultdict(dict, toml.load(f))

    dependencies = parse_poetry_dependencies(pyproject)
    pyproject["project"]["dependencies"] = dependencies
    with open(pyproject_file, "w") as f:
        toml.dump(pyproject, f)
