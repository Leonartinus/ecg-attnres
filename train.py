"""Training entry point.

Usage:
    python train.py --config configs/block_attnres_3.yaml --seed 42
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from models import ECGClassifier, ECGTokenizer, StandardTransformerEncoder, FullAttnResEncoder, BlockAttnResEncoder
from utils.data import load_metadata, load_signals, get_splits, compute_pos_weights, PTBXLDataset
from utils.metrics import compute_macro_auroc


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    """Dispatch on cfg['model']['type'] to build tokenizer + encoder + classifier."""
    tokenizer = ECGTokenizer(n_leads=cfg['tokenizer']['n_leads'], d_model=cfg['tokenizer']['d_model'])
    
    if cfg['model']['type'] == 'standard':
        encoder = StandardTransformerEncoder(
            d_model=cfg['model']['d_model'],
            n_layers=cfg['model']['n_layers'],
            n_heads=cfg['model']['n_heads'],
            d_ff=cfg['model']['d_ff'],
            dropout=cfg['model']['dropout'],
            prenorm=True  # assuming prenorm for standard
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


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Returns dict with val_loss and val_macro_auroc."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()
    macro_auroc = compute_macro_auroc(labels, probs)
    return {
        'val_loss': total_loss / len(loader),
        'val_macro_auroc': macro_auroc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override config with command line args if provided
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_dir = Path(cfg['data']['data_dir'])
    df = load_metadata(data_dir)
    print(f"Loaded metadata for {len(df)} samples.")
    X = load_signals(df, data_dir, sampling_rate=cfg['training']['sampling_rate'])
    print(f"Loaded signals with shape {X.shape}.")
    splits = get_splits(df, X)
    print(f"Train samples: {len(splits['X_train'])}, Val samples: {len(splits['X_val'])}, Test samples: {len(splits['X_test'])}")

    # For testing using a small subset
    # subset_size = 10
    # splits['X_train'] = splits['X_train'][:subset_size]
    # splits['y_train'] = splits['y_train'][:subset_size]
    # splits['X_val'] = splits['X_val'][:subset_size]
    # splits['y_val'] = splits['y_val'][:subset_size]

    # Create datasets
    train_dataset = PTBXLDataset(splits['X_train'], splits['y_train'], augment=cfg['training']['augment'])
    val_dataset = PTBXLDataset(splits['X_val'], splits['y_val'], augment=False)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Seeded generator so shuffle order is reproducible across runs with the same --seed.
    g = torch.Generator()
    g.manual_seed(args.seed)
    print(f"DataLoader generator seed: {args.seed}")

    # num_workers=0: avoids fork copy-on-write blowup on Colab (each worker would
    # duplicate the parent's X array as Python touches refcounts → OOM).
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=0, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=0,
    )

    # Build model
    model = build_model(cfg)
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")

    # Optimizer
    if cfg['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {cfg['training']['optimizer']}")
    
    print(f"Using optimizer: {cfg['training']['optimizer']} with lr={cfg['training']['lr']} and weight_decay={cfg['training']['weight_decay']}")

    # Scheduler
    if cfg['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'], eta_min=cfg['training']['min_lr'])
    else:
        raise ValueError(f"Unsupported scheduler: {cfg['training']['scheduler']}")
    
    print(f"Using scheduler: {cfg['training']['scheduler']} with min_lr={cfg['training']['min_lr']}")

    # Criterion
    pos_weights = compute_pos_weights(splits['y_train'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    # Resume if needed
    start_epoch = 0
    best_auroc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_auroc = checkpoint['best_auroc']

    # Logging
    log_dir = Path(cfg['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{cfg['experiment_name']}_seed{args.seed}.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_macro_auroc'])

    checkpoint_dir = Path(cfg['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{cfg['experiment_name']}_seed{args.seed}_best.pth"
    print(f"Logging to {log_file}, saving best model to {best_checkpoint_path}")

    # Training loop
    patience = cfg['training']['early_stop_patience']
    patience_counter = 0

    for epoch in range(start_epoch, cfg['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Log
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_metrics['val_loss'], val_metrics['val_macro_auroc']])

        print(f"Epoch {epoch + 1}/{cfg['training']['epochs']}: Train Loss {train_loss:.4f}, Val Loss {val_metrics['val_loss']:.4f}, Val AUROC {val_metrics['val_macro_auroc']:.4f}")

        # Save best model
        if val_metrics['val_macro_auroc'] > best_auroc:
            best_auroc = val_metrics['val_macro_auroc']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_auroc': best_auroc
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training complete. Best AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
