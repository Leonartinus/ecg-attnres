"""Training entry point.

Usage:
    python train.py --config configs/block_attnres_3.yaml --seed 42
"""
import argparse
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    """Dispatch on cfg['model']['type'] to build tokenizer + encoder + classifier."""
    # TODO: from models import ECGTokenizer, ECGClassifier
    # TODO: build StandardTransformerEncoder / FullAttnResEncoder / BlockAttnResEncoder
    # TODO: wrap in ECGClassifier
    raise NotImplementedError


def train_one_epoch(model, loader, optimizer, criterion, device):
    raise NotImplementedError


def evaluate(model, loader, criterion, device):
    """Returns dict with val_loss and val_macro_auroc."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # TODO: set seeds (torch, numpy, python random)
    # TODO: load data via utils.data
    # TODO: build model, optimizer, scheduler, criterion (BCEWithLogitsLoss with pos_weight)
    # TODO: training loop with early stopping on val macro AUROC
    # TODO: save best checkpoint to {checkpoint_dir}/{experiment_name}_seed{seed}.pt
    # TODO: save per-epoch metrics to {log_dir}/{experiment_name}_seed{seed}.csv

    raise NotImplementedError


if __name__ == "__main__":
    main()
