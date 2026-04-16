"""Evaluation entry point — loads a checkpoint, runs on test fold, saves predictions.

Usage:
    python evaluate.py --checkpoint checkpoints/block_attnres_3_seed42.pt --config configs/block_attnres_3.yaml
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    # TODO: load config, build model, load checkpoint
    # TODO: load test split (fold 10)
    # TODO: run inference, save preds.npy and targets.npy
    # TODO: compute macro AUROC + bootstrap CI; save to outputs/{experiment_name}_seed{seed}_results.json

    raise NotImplementedError


if __name__ == "__main__":
    main()
