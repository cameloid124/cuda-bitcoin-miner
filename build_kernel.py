#!/usr/bin/env python3
"""Compile kernels/sha256d_mine.cu to PTX (primary) and optional cubin fallbacks."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNELS = ROOT / "kernels"
CU_FILE = KERNELS / "sha256d_mine.cu"
PTX_FILE = KERNELS / "sha256d_mine.ptx"
CUBIN_DIR = KERNELS / "cubin"

# Virtual arch for portable PTX (Turing+). Driver JITs to the actual GPU.
PTX_ARCH = "compute_75"

# Native cubin fallbacks when PTX JIT fails (built alongside PTX in CI/Docker).
# sm_87 = Jetson Orin family (Orin Nano / NX / AGX).
CUBIN_ARCHES = ("sm_75", "sm_86", "sm_87", "sm_89", "sm_90")


def find_nvcc() -> str:
    nvcc = os.environ.get("CUDA_NVCC_EXECUTABLE") or shutil.which("nvcc")
    if nvcc:
        return nvcc
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "nvcc not found. Install the CUDA toolkit or set CUDA_HOME / CUDA_NVCC_EXECUTABLE."
    )


def run_nvcc(nvcc: str, args: list[str]) -> None:
    cmd = [nvcc, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if not CU_FILE.is_file():
        raise SystemExit(f"Missing kernel source: {CU_FILE}")

    nvcc = find_nvcc()
    common = ["-O3", "--std=c++17", str(CU_FILE)]

    run_nvcc(nvcc, ["-ptx", f"-arch={PTX_ARCH}", *common, "-o", str(PTX_FILE)])
    print(f"Wrote {PTX_FILE} ({PTX_FILE.stat().st_size} bytes)")

    CUBIN_DIR.mkdir(parents=True, exist_ok=True)
    for arch in CUBIN_ARCHES:
        out = CUBIN_DIR / f"{arch}.cubin"
        run_nvcc(nvcc, ["-cubin", f"-arch={arch}", *common, "-o", str(out)])
        print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
