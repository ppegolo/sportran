# -*- coding: utf-8 -*-
"""Check that the GitHub release tag matches the package version."""

import argparse
import re


def _read_project_version(pyproject_path):
    in_project_section = False
    version_re = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']\s*$')
    with open(pyproject_path, encoding="utf8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("["):
                in_project_section = line == "[project]"
                continue
            if in_project_section:
                match = version_re.match(line)
                if match:
                    return match.group(1)
    raise ValueError(f"Cannot find [project].version in {pyproject_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("GITHUB_REF", help="The GITHUB_REF environmental variable")
    parser.add_argument("PYPROJECT_PATH", help="Path to pyproject.toml")
    args = parser.parse_args()
    assert args.GITHUB_REF.startswith("refs/tags/v"), f'GITHUB_REF should start with "refs/tags/v": {args.GITHUB_REF}'
    tag_version = args.GITHUB_REF[11:]
    pypi_version = _read_project_version(args.PYPROJECT_PATH)
    assert tag_version == pypi_version, f"The tag version {tag_version} != {pypi_version} specified in `pyproject.toml`"
