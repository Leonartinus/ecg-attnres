#!/bin/bash
# Generate attention-weight visualizations for every attn-res checkpoint × seed.
# Outputs go to figures/<config>_seed<seed>/
#
# NOTE: analysis/attention_weights.py currently hardcodes FullAttnResEncoder in
# load_model(). To run block_attnres_* configs, extend that function to dispatch
# on cfg["encoder"] (or similar) before using this script.

set -e

SEEDS=(42 43 44 45 46)
CONFIGS=(
    configs/full_attnres.yaml
    configs/block_attnres_2.yaml
    configs/block_attnres_3.yaml
)
DATA_DIR="${DATA_DIR:-data/ptbxl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-figures}"
BATCH_SIZE="${BATCH_SIZE:-64}"

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)

    for seed in "${SEEDS[@]}"; do
        ckpt="checkpoints/${name}_seed${seed}_best.pth"
        out_dir="${OUTPUT_ROOT}/${name}_seed${seed}"

        if [ ! -f "$ckpt" ]; then
            echo "[skip] missing checkpoint: $ckpt"
            continue
        fi

        echo "=== Visualizing $name (seed $seed) -> $out_dir ==="
        python analysis/attention_weights.py \
            --checkpoint "$ckpt" \
            --config "$config" \
            --data_dir "$DATA_DIR" \
            --output_dir "$out_dir" \
            --batch_size "$BATCH_SIZE"
    done
done
