"""Manage one run's deadline, stop signals, and checkpoint schedule."""

import signal
import time

from ..config import TrainingConfig


class TrainingStopped(Exception):
    pass


class RunControl:
    def __init__(self, settings: TrainingConfig):
        self.settings = settings
        self.started = time.monotonic()
        self.last_checkpoint = self.started
        self.stop_reason: str | None = None
        self.previous_handlers = {}

    def __enter__(self):
        for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signal_number = getattr(signal, signal_name, None)
            if signal_number is not None:
                self.previous_handlers[signal_number] = signal.signal(
                    signal_number, self.request_stop
                )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for signal_number, handler in self.previous_handlers.items():
            signal.signal(signal_number, handler)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def stop_requested(self) -> bool:
        return self.stop_reason is not None

    @property
    def checkpoint_due(self) -> bool:
        return (
            time.monotonic() - self.last_checkpoint
            >= self.settings.checkpoint_every_seconds
        )

    def checkpoint_saved(self) -> None:
        self.last_checkpoint = time.monotonic()

    def request_stop(self, signum, frame) -> None:
        self.stop_reason = "interrupted"

    def poll(self) -> None:
        if (
            self.stop_reason is None
            and self.elapsed_seconds >= self.settings.max_runtime_seconds
        ):
            self.stop_reason = "runtime_limit"

    def raise_if_stopped(self) -> None:
        self.poll()
        if self.stop_requested:
            raise TrainingStopped()
