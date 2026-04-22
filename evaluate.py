"""Evaluation entry point — loads a checkpoint, runs on test fold, saves predictions.

Usage:
    python evaluate.py --checkpoint checkpoints/block_attnres_3_seed42.pt --config configs/block_attnres_3.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import ECGClassifier, ECGTokenizer, StandardTransformerEncoder, FullAttnResEncoder, BlockAttnResEncoder
from utils.data import load_metadata, load_signals, get_splits, PTBXLDataset
from utils.metrics import compute_macro_auroc, compute_f_max, compute_bootstrap_ci
from utils.viz import plot_depth_attention_heatmap


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    """Same as in train.py"""
    tokenizer = ECGTokenizer(n_leads=cfg['tokenizer']['n_leads'], d_model=cfg['tokenizer']['d_model'])
    
    if cfg['model']['type'] == 'standard':
        encoder = StandardTransformerEncoder(
            d_model=cfg['model']['d_model'],
            n_layers=cfg['model']['n_layers'],
            n_heads=cfg['model']['n_heads'],
            d_ff=cfg['model']['d_ff'],
            dropout=cfg['model']['dropout'],
            prenorm=True
        )
    elif cfg['model']['type'] == 'full_attnres':
        encoder = FullAttnResEncoder(
            d_model=cfg['model']['d_model'],
            n_layers=cfg['model']['n_layers'],
            n_heads=cfg['model']['n_heads'],
            d_ff=cfg['model']['d_ff'],
            dropout=cfg['model']['dropout']
        )
    elif cfg['model']['type'] == 'block_attnres':
        encoder = BlockAttnResEncoder(
            d_model=cfg['model']['d_model'],
            n_blocks=cfg['model']['n_blocks'],
            n_layers=cfg['model']['n_layers'],
            n_heads=cfg['model']['n_heads'],
            d_ff=cfg['model']['d_ff'],
            dropout=cfg['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {cfg['model']['type']}")
    
    demographic_dim = 0  # No demographics in data
    model = ECGClassifier(
        tokenizer=tokenizer,
        encoder=encoder,
        d_model=cfg['model']['d_model'],
        n_classes=cfg['head']['n_classes'],
        demographic_dim=demographic_dim,
        dropout=cfg['head']['dropout']
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_dir = Path('../data/')
    df = load_metadata(data_dir)
    X = load_signals(df, data_dir, sampling_rate=cfg['training']['sampling_rate'])
    splits = get_splits(df, X)

    # Test dataset
    test_dataset = PTBXLDataset(splits['X_test'], splits['y_test'], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)

    # Build model
    model = build_model(cfg)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Run inference
    all_logits = []
    all_labels = []
    all_alphas = []  # if AttnRes
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            # If AttnRes, collect alphas
            if hasattr(model.encoder, '_last_alphas') and model.encoder._last_alphas is not None:
                all_alphas.append(model.encoder._last_alphas.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()

    # Compute metrics
    macro_auroc = compute_macro_auroc(labels, probs)
    f_max = compute_f_max(labels, probs)
    ci = compute_bootstrap_ci(labels, probs, n_bootstraps=1000)

    results = {
        'macro_auroc': macro_auroc,
        'f_max': f_max,
        'ci': ci
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / f"{cfg['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print(f"Macro AUROC: {macro_auroc:.4f}")
    print(f"F_max: {f_max:.4f}")
    print(f"CI AUROC: {ci['auroc']}")
    print(f"CI F_max: {ci['f_max']}")

    # Plot attention weights if available
    if all_alphas:
        # Average alphas over samples
        alphas = torch.stack(all_alphas, dim=0).mean(dim=0).numpy()  # (n_layers, n_prev_layers) or something
        # For FullAttnRes, alphas shape is (batch, n_layers, n_prev)
        # But averaged: (n_layers, n_prev)
        # For plot, perhaps per layer, attention to previous
        # But the plot is (n_classes, n_layers), but alphas are not per class
        # The user said "Attention Residual weights (α) from the attnres.py layers to show which ECG features the model prioritized"
        # Perhaps plot the alphas as heatmap, rows layers, columns previous layers
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']  # but not per class
        layer_names = [f'Layer {i}' for i in range(alphas.shape[0])]
        plot_path = output_dir / f"{cfg['experiment_name']}_attention_heatmap.png"
        plot_depth_attention_heatmap(alphas, class_names, layer_names, save_path=str(plot_path))
        print(f"Attention heatmap saved to {plot_path}")
    else:
        print("No attention weights available (not AttnRes model)")


if __name__ == "__main__":
    main()