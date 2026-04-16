#!/bin/bash
# Run all model variants across 5 seeds.
# Expects train.py to be working.

set -e
SEEDS=(42 43 44 45 46)
CONFIGS=(
    configs/baseline_prenorm.yaml
    configs/baseline_postnorm.yaml
    configs/full_attnres.yaml
    configs/block_attnres_3.yaml
    configs/block_attnres_2.yaml
)

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running $config with seed $seed"
        python train.py --config "$config" --seed "$seed"
    done
done
