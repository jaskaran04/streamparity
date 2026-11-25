# Release checklist (maintainer)

1. Confirm the package name is available/claimable on both PyPI and TestPyPI. Verify the MIT licence attribution and repository ownership. Do not claim public availability until upload succeeds.
2. Create a GitHub repository, push the contents, and wait for Linux/Windows CI to pass. Add actual repository URLs to `pyproject.toml` when known. No credentials belong in git.
3. From the repository root in a fresh virtual environment:
   ```sh
   python -m pip install -e ".[dev]"
   python -m pytest -q
   python examples/trading_replay.py
   python -m build
   python -m twine check dist/*
   ```
4. Create and verify TestPyPI and PyPI accounts and configure multifactor authentication. Use a TestPyPI-scoped API token for a manual test upload; enter it at the prompt, not in code or chat:
   ```sh
   python -m twine upload --repository testpypi dist/*
   ```
5. In a separate environment, install NumPy from regular PyPI and then install the exact test release without dependency resolution:
   ```sh
   python -m pip install "numpy>=1.24"
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps streamparity==0.2.0
   python -c "import streamparity; print(streamparity.__all__)"
   ```
6. For public publishing, configure a PyPI Trusted Publisher for the repository and matching workflow/environment (see https://docs.pypi.org/trusted-publishers/). Alternatively, use an appropriately scoped PyPI API token and `python -m twine upload dist/*` after explicitly approving release. PyPI and TestPyPI use separate credentials. Never reuse already uploaded filenames for a modified release; increment the version and rebuild from a clean tree.
7. Verify the actual public installation in a clean environment with `python -m pip install streamparity==0.2.0` and rerun a small smoke test. Tag the corresponding Git commit and publish release notes.
