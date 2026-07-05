"""Minimal CUDA Driver API bindings via ctypes (no CuPy / NumPy)."""
from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Types and constants (Driver API)
# ---------------------------------------------------------------------------

CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUdeviceptr = ctypes.c_uint64

# Flags=0 creates legacy-blocking streams: they synchronize with operations on
# the default stream (0). GpuScanner relies on this so that synchronous
# job/target uploads on the default stream are ordered against in-flight
# kernels on the worker streams (matching CuPy's default Stream semantics).
CU_STREAM_DEFAULT = 0x0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
PTX_PATH = KERNELS_DIR / "sha256d_mine.ptx"
CUBIN_DIR = KERNELS_DIR / "cubin"


class KernelLoadInfo(NamedTuple):
    """Describes which kernel artifact was loaded into the CUDA driver."""

    kind: str       # "ptx" or "cubin"
    artifact: str   # repo-relative path, e.g. kernels/sha256d_mine.ptx


class CUDAError(Exception):
    """Raised when a CUDA Driver API call returns a non-success code."""


def _load_cuda_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda.dll")
    return ctypes.CDLL("libcuda.so.1")


class CudaDriver:
    """Singleton wrapper around the CUDA Driver API for one GPU."""

    _instance: CudaDriver | None = None

    @classmethod
    def instance(cls) -> CudaDriver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lib = _load_cuda_library()
        self._bind()
        self._check(self._lib.cuInit(0), "cuInit")

        device = CUdevice()
        self._check(self._lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        self._device = device.value

        ctx = ctypes.c_void_p()
        err = self._lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, self._device)
        if err == 704:  # CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
            self._check(
                self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self._device),
                "cuDevicePrimaryCtxRetain",
            )
        else:
            self._check(err, "cuCtxCreate_v2")
        self._ctx = ctx
        self._ctx_bound = threading.local()
        self._bind_context()

        self._kernel_load_info: KernelLoadInfo | None = None
        self._module = self._load_mining_module()
        func = ctypes.c_void_p()
        self._check(
            self._lib.cuModuleGetFunction(ctypes.byref(func), self._module, b"sha256d_mine"),
            "cuModuleGetFunction",
        )
        self._mine_function = func

    @property
    def kernel_load_info(self) -> KernelLoadInfo:
        if self._kernel_load_info is None:
            raise CUDAError("Mining kernel was not loaded.")
        return self._kernel_load_info

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def _load_mining_module(self) -> ctypes.c_void_p:
        errors: list[str] = []

        if PTX_PATH.is_file():
            try:
                module = self._load_module_bytes(PTX_PATH.read_bytes(), text=True)
                self._kernel_load_info = KernelLoadInfo("ptx", self._rel_artifact(PTX_PATH))
                return module
            except CUDAError as exc:
                errors.append(f"PTX: {exc}")

        major, minor = self.get_compute_capability()
        cubin_path = CUBIN_DIR / f"sm_{major}{minor}.cubin"
        if cubin_path.is_file():
            try:
                module = self._load_module_bytes(cubin_path.read_bytes(), text=False)
                self._kernel_load_info = KernelLoadInfo("cubin", self._rel_artifact(cubin_path))
                return module
            except CUDAError as exc:
                errors.append(f"cubin ({cubin_path.name}): {exc}")
        else:
            errors.append(f"cubin: {cubin_path} not found")

        hint = "Run: python build_kernel.py"
        detail = "; ".join(errors) if errors else "no kernel artifacts present"
        raise CUDAError(
            f"Could not load mining kernel ({detail}). {hint}"
        )

    @staticmethod
    def _rel_artifact(path: Path) -> str:
        try:
            return path.relative_to(KERNELS_DIR.parent).as_posix()
        except ValueError:
            return path.as_posix()

    def _load_module_bytes(self, data: bytes, *, text: bool) -> ctypes.c_void_p:
        module = ctypes.c_void_p()
        payload = (data + b"\0") if text else data
        buf = ctypes.c_char_p(payload)
        self._check(self._lib.cuModuleLoadData(ctypes.byref(module), buf), "cuModuleLoadData")
        return module

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    def get_device_name(self) -> str:
        self._bind_context()
        name = ctypes.create_string_buffer(256)
        self._check(self._lib.cuDeviceGetName(name, 256, self._device), "cuDeviceGetName")
        return name.value.decode()

    def get_driver_version(self) -> int:
        self._bind_context()
        version = ctypes.c_int()
        self._check(self._lib.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
        return version.value

    def get_compute_capability(self) -> tuple[int, int]:
        self._bind_context()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self._check(
            self._lib.cuDeviceGetAttribute(
                ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self._device),
            "cuDeviceGetAttribute(major)",
        )
        self._check(
            self._lib.cuDeviceGetAttribute(
                ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self._device),
            "cuDeviceGetAttribute(minor)",
        )
        return major.value, minor.value

    # ------------------------------------------------------------------
    # Memory and streams
    # ------------------------------------------------------------------

    def mem_alloc(self, size: int) -> int:
        self._bind_context()
        ptr = CUdeviceptr()
        self._check(self._lib.cuMemAlloc_v2(ctypes.byref(ptr), size), "cuMemAlloc_v2")
        return ptr.value

    def mem_free(self, ptr: int) -> None:
        self._bind_context()
        self._check(self._lib.cuMemFree_v2(ptr), "cuMemFree_v2")

    def mem_host_alloc(self, size: int) -> int:
        self._bind_context()
        ptr = ctypes.c_void_p()
        self._check(self._lib.cuMemHostAlloc(ctypes.byref(ptr), size, 0), "cuMemHostAlloc")
        return ptr.value

    def mem_free_host(self, ptr: int) -> None:
        self._bind_context()
        self._check(self._lib.cuMemFreeHost(ptr), "cuMemFreeHost")

    def create_stream(self) -> int:
        self._bind_context()
        stream = ctypes.c_void_p()
        self._check(
            self._lib.cuStreamCreate(ctypes.byref(stream), CU_STREAM_DEFAULT),
            "cuStreamCreate",
        )
        return stream.value

    def stream_sync(self, stream: int) -> None:
        self._bind_context()
        self._check(self._lib.cuStreamSynchronize(ctypes.c_void_p(stream)), "cuStreamSynchronize")

    def memset_d32_async(self, ptr: int, value: int, count: int, stream: int) -> None:
        self._bind_context()
        self._check(
            self._lib.cuMemsetD32Async(ptr, value, count, ctypes.c_void_p(stream)),
            "cuMemsetD32Async",
        )

    def memcpy_dtoh_async(self, dst: int, src: int, size: int, stream: int) -> None:
        self._bind_context()
        self._check(
            self._lib.cuMemcpyDtoHAsync_v2(ctypes.c_void_p(dst), src, size, ctypes.c_void_p(stream)),
            "cuMemcpyDtoHAsync_v2",
        )

    def memcpy_htod(self, dst: int, src: int, size: int) -> None:
        """Host-to-device copy on the legacy default stream (stream 0)."""
        self._bind_context()
        self._check(
            self._lib.cuMemcpyHtoD_v2(dst, ctypes.c_void_p(src), size),
            "cuMemcpyHtoD_v2",
        )

    def memcpy_dtoh(self, dst: int, src: int, size: int) -> None:
        self._bind_context()
        self._check(
            self._lib.cuMemcpyDtoH_v2(ctypes.c_void_p(dst), src, size),
            "cuMemcpyDtoH_v2",
        )

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------

    def launch_mine(
        self,
        stream: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        d_job: int,
        nonce_base: int,
        d_target: int,
        d_results: int,
        d_best_hash: int,
        d_best_msb: int,
    ) -> None:
        self._bind_context()
        d_job_arg = CUdeviceptr(d_job)
        nonce_arg = ctypes.c_uint32(nonce_base & 0xFFFFFFFF)
        d_target_arg = CUdeviceptr(d_target)
        d_results_arg = CUdeviceptr(d_results)
        d_best_hash_arg = CUdeviceptr(d_best_hash)
        d_best_msb_arg = CUdeviceptr(d_best_msb)

        params = (ctypes.c_void_p * 6)(
            ctypes.addressof(d_job_arg),
            ctypes.addressof(nonce_arg),
            ctypes.addressof(d_target_arg),
            ctypes.addressof(d_results_arg),
            ctypes.addressof(d_best_hash_arg),
            ctypes.addressof(d_best_msb_arg),
        )

        self._check(
            self._lib.cuLaunchKernel(
                self._mine_function,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                0, ctypes.c_void_p(stream),
                ctypes.cast(params, ctypes.c_void_p), None,
            ),
            "cuLaunchKernel",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bind_context(self) -> None:
        """Attach the CUDA context to the calling thread.

        Contexts are thread-local. asyncio.to_thread() runs GpuScanner on pool
        threads that never saw cuCtxSetCurrent, which surfaces as error 201
        (CUDA_ERROR_INVALID_CONTEXT) on the first memcpy/launch.
        """
        if getattr(self._ctx_bound, "active", False):
            return
        self._check(self._lib.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
        self._ctx_bound.active = True

    def _check(self, result: int, name: str) -> None:
        if result != 0:
            err_name = ctypes.c_char_p()
            if self._lib.cuGetErrorName(result, ctypes.byref(err_name)) == 0 and err_name.value:
                detail = err_name.value.decode()
            else:
                detail = "unknown error"
            raise CUDAError(f"{name} failed with error {result} ({detail})")

    def _bind(self) -> None:
        L = self._lib

        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuInit.restype = CUresult

        L.cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        L.cuGetErrorName.restype = CUresult

        L.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
        L.cuDeviceGet.restype = CUresult

        L.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, CUdevice]
        L.cuDeviceGetName.restype = CUresult

        L.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, CUdevice]
        L.cuDeviceGetAttribute.restype = CUresult

        L.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, CUdevice]
        L.cuCtxCreate_v2.restype = CUresult

        L.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), CUdevice]
        L.cuDevicePrimaryCtxRetain.restype = CUresult

        L.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        L.cuCtxSetCurrent.restype = CUresult

        L.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.cuDriverGetVersion.restype = CUresult

        L.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        L.cuModuleLoadData.restype = CUresult

        L.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p,
        ]
        L.cuModuleGetFunction.restype = CUresult

        L.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
        L.cuMemAlloc_v2.restype = CUresult

        L.cuMemFree_v2.argtypes = [CUdeviceptr]
        L.cuMemFree_v2.restype = CUresult

        L.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        L.cuMemHostAlloc.restype = CUresult

        L.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        L.cuMemFreeHost.restype = CUresult

        L.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        L.cuStreamCreate.restype = CUresult

        L.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        L.cuStreamSynchronize.restype = CUresult

        L.cuMemsetD32Async.argtypes = [CUdeviceptr, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
        L.cuMemsetD32Async.restype = CUresult

        L.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
        L.cuMemcpyHtoD_v2.restype = CUresult

        L.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
        L.cuMemcpyDtoH_v2.restype = CUresult

        L.cuMemcpyDtoHAsync_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t, ctypes.c_void_p]
        L.cuMemcpyDtoHAsync_v2.restype = CUresult

        L.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        L.cuLaunchKernel.restype = CUresult
