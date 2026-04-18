"""Execute a notebook in-place with a custom kernel pointing at conda python.

Usage:
    python scripts/execute_notebook.py notebooks/03_panel_evaluation.ipynb \
        notebooks/03_panel_evaluation.executed.ipynb
"""
from __future__ import annotations

import os
import sys

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(ROOT, ".kernels")


def main(src: str, dst: str) -> None:
    os.environ["JUPYTER_PATH"] = KERNEL_DIR + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    ksm = KernelSpecManager()
    ksm.kernel_dirs = [KERNEL_DIR] + list(ksm.kernel_dirs)
    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="pml-conda",
        kernel_spec_manager=ksm,
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(src))}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(nb, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    main(src, dst)
