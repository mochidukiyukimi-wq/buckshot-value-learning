from dataclasses import dataclass, asdict
import platform
import sys

import torch


@dataclass
class Metrics:
    search_seconds: float = 0
    inference_seconds: float = 0
    transfer_seconds: float = 0
    update_seconds: float = 0
    inference_rows: int = 0
    inference_batches: int = 0
    internal_nodes: int = 0
    frontier_nodes: int = 0
    cache_hits: int = 0
    transitions: int = 0
    terminal_transitions: int = 0
    boundary_transitions: int = 0
    search_memory_estimate_bytes: int = 0
    root_count: int = 0
    teacher_rows: int = 0
    updates: int = 0
    rows_used: int = 0
    resource_limit: int = 0
    incomplete_roots: int = 0
    invalid: int = 0
    nan: int = 0

    def record_search(self, stats) -> None:
        for name in (
            "internal_nodes",
            "frontier_nodes",
            "cache_hits",
            "transitions",
            "terminal_transitions",
            "boundary_transitions",
        ):
            setattr(self, name, getattr(self, name) + getattr(stats, name))
        self.search_memory_estimate_bytes = max(
            self.search_memory_estimate_bytes, stats.approximate_memory_bytes
        )

    def record_inference(
        self, rows: int, seconds: float, transfer_seconds: float
    ) -> None:
        self.inference_rows += rows
        self.inference_batches += 1
        self.inference_seconds += seconds
        self.transfer_seconds += transfer_seconds

    def record_update(self, rows: int, seconds: float) -> None:
        self.updates += 1
        self.rows_used += rows
        self.update_seconds += seconds

    def summarize_interval(self) -> dict:
        values = asdict(self)
        values["cache_hit_rate"] = self.cache_hits / max(
            1, self.cache_hits + self.internal_nodes + self.frontier_nodes
        )
        values["mean_inference_batch_rows"] = self.inference_rows / max(
            1, self.inference_batches
        )
        values["inference_rows_per_second"] = self.inference_rows / max(
            1e-9, self.inference_seconds
        )
        values["teacher_reuse"] = self.rows_used / max(1, self.teacher_rows)
        values["process_peak_rss_bytes"] = process_peak_rss_bytes()
        return values


def process_peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


def execution_environment(config) -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": config.device,
        "cuda_available": torch.cuda.is_available(),
        "dtype": "float32",
        "intra_threads": torch.get_num_threads(),
        "inter_threads": torch.get_num_interop_threads(),
        "compile": False,
        "mixed_precision": False,
        "search_workers": 1,
        "waiting_queue": "one request per root",
        "asynchronous_transfer": False,
    }
