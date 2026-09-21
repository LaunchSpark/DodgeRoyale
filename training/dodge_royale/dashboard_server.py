"""Start marimo after PyTorch has finished initializing.

Marimo run mode shares import hooks across its kernel threads. Concurrent
sessions can otherwise import PyTorch while its formatter is registering and
see a partially initialized ``torch`` module, taking every dashboard cell
down with the first failed import.
"""

from __future__ import annotations

import runpy


def main() -> None:
    import torch.nn  # noqa: F401 - finish this import before marimo installs hooks

    runpy.run_module("marimo", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
