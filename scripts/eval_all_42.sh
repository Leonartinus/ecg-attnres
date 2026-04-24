#!/bin/bash
# Evaluate all 5 model variants at seed 42 on the test fold.
# Expects checkpoints/{experiment_name}_seed42_best.pth to exist for each config.

set -e
SEED=42
CONFIGS=(
    configs/baseline_postnorm.yaml
    configs/baseline_prenorm.yaml
    configs/full_attnres.yaml
    configs/block_attnres_2.yaml
    configs/block_attnres_3.yaml
)

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    ckpt="checkpoints/${name}_seed${SEED}_best.pth"

    if [ ! -f "$ckpt" ]; then
        echo "[skip] missing checkpoint: $ckpt"
        continue
    fi

    echo "=== Evaluating $name (seed $SEED) ==="
    python evaluate.py --checkpoint "$ckpt" --config "$config"
done
