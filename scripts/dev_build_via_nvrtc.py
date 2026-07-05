#!/usr/bin/env python3
"""Fallback PTX/cubin build using NVRTC (CuPy). Dev-only — not a runtime dependency."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"
CU_FILE = KERNELS / "sha256d_mine.cu"
PTX_FILE = KERNELS / "sha256d_mine.ptx"
CUBIN_DIR = KERNELS / "cubin"
PTX_ARCH = "75"
# sm_87 = Jetson Orin family (Orin Nano / NX / AGX).
CUBIN_ARCHES = ("75", "86", "87", "89", "90")


def main() -> None:
    try:
        from cupy.cuda.compiler import _NVRTCProgram
    except ImportError as exc:
        raise SystemExit(
            "CuPy is required for this fallback builder. "
            "Install the CUDA toolkit and run build_kernel.py instead."
        ) from exc

    src = CU_FILE.read_text(encoding="utf-8")
    CUBIN_DIR.mkdir(parents=True, exist_ok=True)

    prog = _NVRTCProgram(src, "sha256d_mine.cu", method="ptx")
    ptx, _ = prog.compile(("--std=c++17", f"-arch=compute_{PTX_ARCH}"))
    PTX_FILE.write_bytes(ptx)
    print(f"Wrote {PTX_FILE} ({len(ptx)} bytes)")

    for arch in CUBIN_ARCHES:
        prog = _NVRTCProgram(src, "sha256d_mine.cu", method="cubin")
        cubin, _ = prog.compile(("--std=c++17", f"-arch=sm_{arch}"))
        out = CUBIN_DIR / f"sm_{arch}.cubin"
        out.write_bytes(cubin)
        print(f"Wrote {out} ({len(cubin)} bytes)")


if __name__ == "__main__":
    main()
