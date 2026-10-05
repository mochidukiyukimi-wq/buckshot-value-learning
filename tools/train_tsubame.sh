#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/tsubame

resume_option=${1:-}
if [[ $# -gt 1 || ( -n "$resume_option" && "$resume_option" != --ema-decay-transition && "$resume_option" != --loss-transition ) ]]; then
    echo "Usage: tools/train_tsubame.sh [--ema-decay-transition|--loss-transition]" >&2
    exit 2
fi

# Keep five minutes for an orderly stop and final write before scheduler termination.
allocation_deadline=${BUCKSHOT_ALLOCATION_DEADLINE:?Set the allocated job deadline in Unix seconds}
resume_mode=$(.venv/bin/python - "$allocation_deadline" "$resume_option" <<'PY'
import json
from pathlib import Path
import sys
import time
import torch
from roulette.config import load_config
from roulette.training.checkpoint import validate_checkpoint_metadata

configuration = json.loads(Path('configs/tsubame.json').read_text())
remaining_seconds = int(sys.argv[1]) - time.time() - 300
if remaining_seconds < 60:
    raise RuntimeError('Allocation has insufficient time for a safe training launch')
configuration['training']['max_runtime_seconds'] = remaining_seconds
Path('runs/tsubame/launch_config.json').write_text(json.dumps(configuration, indent=2))
print('TRAINING_RUNTIME_SECONDS', remaining_seconds, file=sys.stderr, flush=True)
config = load_config('runs/tsubame/launch_config.json')
saved = torch.load('runs/tsubame/latest.pt', map_location='cpu', weights_only=False)
try:
    validate_checkpoint_metadata(saved['metadata'], config)
except ValueError:
    if sys.argv[2] == '--loss-transition':
        validate_checkpoint_metadata(saved['metadata'], config, allow_loss_change=True,
                                     allow_loss_learning_rate_change=True)
        print('loss-transition')
    elif sys.argv[2] == '--ema-decay-transition':
        validate_checkpoint_metadata(saved['metadata'], config, allow_ema_decay_change=True)
        print('ema-decay-transition')
    else:
        # Only the imported CPU checkpoint may enter a new stage automatically.
        if saved['metadata']['config']['device'] != 'cpu':
            raise
        print('stage-transition')
else:
    print('resume')
PY
)

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# The wrapper names any permitted transition; routine continuation stays strict.
resume_arguments=(--resume)
if [[ "$resume_mode" != resume ]]; then
    resume_arguments+=("--$resume_mode")
fi
.venv/bin/python -m roulette train --config runs/tsubame/launch_config.json "${resume_arguments[@]}"
