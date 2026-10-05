import json
from pathlib import Path
import time


class RunLogger:
    def __init__(self, run_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.directory = Path(run_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl = (self.directory / "metrics.jsonl").open("a", encoding="utf-8")
        self.tensorboard = SummaryWriter(str(self.directory / "tensorboard"))

    def write_jsonl(self, event: dict) -> None:
        self.jsonl.write(
            json.dumps({"timestamp": time.time(), **event}, allow_nan=False) + "\n"
        )
        self.jsonl.flush()

    def write_tensorboard(self, metrics: dict, step: int, prefix="training") -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tensorboard.add_scalar(f"{prefix}/{name}", value, step)
        self.tensorboard.flush()

    def close(self) -> None:
        self.tensorboard.close()
        self.jsonl.close()


def render_cli(event: dict) -> None:
    validation = event.get("validation", {})
    print(
        f"unit={event['generation_unit']:3d} step={event['step']:4d} "
        f"elapsed={event['elapsed_seconds']:.1f}s "
        f"MAE={validation.get('mae', float('nan')):.5f} "
        f"P99={validation.get('p99', float('nan')):.5f} "
        f"val_step={validation.get('model_version', 0)} "
        f"CE={event.get('cross_entropy', 0):.4f} "
        f"MSE={event.get('value_mse', 0):.5f} "
        f"roots={event['root_count']} labels={event['teacher_rows']} "
        f"rows/s={event.get('inference_rows_per_second', 0):.0f}",
        flush=True,
    )
