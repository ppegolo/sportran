# Contributing

Run commands from repository root.

- Format and autofix: `tox -e format`
- Lint and type checks: `tox -e lint`
- Tests: `tox -e py310` (also `py311`, `py312`)
- Docs build: `tox -e docs`
- Package build and metadata checks: `tox -e build`

Before opening PR, run at least:

- `tox -e lint`
- `tox -e py310`
- `tox -e docs`
- `tox -e build`

Release workflow runs same gates plus publish on version tags.
