"""Root conftest, present so `training/` is importable from the tests.

`tools/` is a directory of scripts rather than a package inside
`dodge_royale`, so `tests/test_benchmark.py` can only import it if this
directory is on `sys.path`. pytest puts the directory holding the topmost
conftest there, which is what this file is for.
"""
