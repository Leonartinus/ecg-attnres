"""Extract and visualize depth-wise attention weights from AttnRes models.

Produces:
    1. Heatmap: conditions x layers showing mean attention weight
    2. Line plots: per-condition attention distribution across depth
    3. Raw numpy arrays saved for further analysis

Usage:
    python analysis/attention_weights.py \
        --checkpoint checkpoints/full_attnres_seed42.pt \
        --config configs/full_attnres.yaml \
        --data_dir /content/drive/MyDrive/ecg-attnres/data/ptbxl \
        --output_dir figures/
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from models import ECGClassifier, ECGTokenizer, FullAttnResEncoder, BlockAttnResEncoder
from utils.data import load_metadata, aggregate_superclass, load_signals, get_splits, PTBXLDataset


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Build model from config and load trained weights."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # TODO: build model from config (same logic as train.py)
    model = ECGClassifier(
        tokenizer=ECGTokenizer(n_leads=cfg["n_leads"], d_model=cfg["d_model"]),
        encoder=FullAttnResEncoder(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"]
        ),
        d_model=cfg["d_model"],
        n_classes=cfg["n_classes"],
        demographic_dim=cfg.get("demographic_dim", 0),
        dropout=cfg.get("dropout", 0.3)
    ).to(device)

    # TODO: model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # TODO: model.eval()
    model.eval()

    return model


def extract_alphas(model, dataloader, device):
    """Run inference on all samples, collect per-sample attention weights.

    Returns:
        all_alphas: list of length n_samples, each entry is a list of alpha tensors
                    from model.encoder._last_alphas
        all_labels: (n_samples, 5) numpy array of binary labels
        all_preds:  (n_samples, 5) numpy array of predicted probabilities
    """
    all_alphas = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.sigmoid(logits).cpu().numpy()

            # Grab cached alphas from the encoder
            # _last_alphas is a list of tensors, one per sublayer call
            # Each tensor shape: (n_values, batch, seq_len)
            batch_alphas = model.encoder._last_alphas

            # Average over seq_len dimension — we care about which DEPTH
            # the layer attends to, not which token position
            # After averaging: each alpha is (n_values, batch)
            batch_alphas_avg = [
                alpha.mean(dim=-1).cpu()  # (n_values, batch)
                for alpha in batch_alphas
            ]

            # Split batch dimension into individual samples
            batch_size = batch_x.shape[0]
            for b in range(batch_size):
                sample_alphas = [alpha[:, b] for alpha in batch_alphas_avg]  # list of (n_values,)
                all_alphas.append(sample_alphas)

            all_labels.append(batch_y.numpy())
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels, axis=0)  # (n_samples, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    return all_alphas, all_labels, all_preds


def aggregate_by_condition(all_alphas, all_labels):
    """Group attention weights by diagnostic condition and average.

    Since PTB-XL is multi-label, a sample with MI+STTC contributes
    to both the MI group and the STTC group.

    Returns:
        condition_alphas: dict mapping condition name to averaged alpha structure
        sample_counts: dict mapping condition name to number of samples
    """
    condition_alphas = {}
    sample_counts = {}

    for c, class_name in enumerate(SUPERCLASSES):
        # Find all samples with this condition
        mask = all_labels[:, c] == 1
        indices = np.where(mask)[0]
        sample_counts[class_name] = len(indices)

        if len(indices) == 0:
            continue

        # For each sublayer's alphas, average across all samples with this condition
        n_sublayers = len(all_alphas[0])
        avg_alphas = []

        for sublayer_idx in range(n_sublayers):
            # Stack this sublayer's alphas for all matching samples
            # Each sample's alpha for this sublayer: (n_values,)
            # But n_values can differ across sublayers (grows with depth)
            stacked = torch.stack([all_alphas[i][sublayer_idx] for i in indices])  # (n_matching, n_values)
            avg_alphas.append(stacked.mean(dim=0).numpy())  # (n_values,)

        condition_alphas[class_name] = avg_alphas

    return condition_alphas, sample_counts


def build_heatmap_matrix(condition_alphas):
    """Convert per-condition alpha lists into a 2D matrix for plotting.

    Strategy: for each condition and each sublayer, take the alpha weight
    assigned to the MOST RECENT value (i.e., how much does this sublayer
    rely on the immediately preceding output vs. earlier ones).

    Alternative strategy: for each sublayer, show the full distribution.
    We implement the simpler version first.

    Returns:
        matrix: (n_conditions, n_sublayers) numpy array
        row_labels: list of condition names
        col_labels: list of sublayer names
    """
    row_labels = []
    rows = []

    for class_name in SUPERCLASSES:
        if class_name not in condition_alphas:
            continue

        alphas = condition_alphas[class_name]
        # For each sublayer, extract the weight on the last (most recent) value
        # This shows "how much does this sublayer rely on the newest information"
        row = [a[-1] for a in alphas]  # last alpha value per sublayer
        rows.append(row)
        row_labels.append(class_name)

    matrix = np.array(rows)  # (n_conditions, n_sublayers)

    # Label sublayers: attn_0, ffn_0, attn_1, ffn_1, ..., final
    n_sublayers = matrix.shape[1]
    col_labels = []
    for i in range((n_sublayers - 1) // 2):
        col_labels.extend([f"L{i}_attn", f"L{i}_ffn"])
    col_labels.append("final")

    return matrix, row_labels, col_labels


def plot_heatmap(matrix, row_labels, col_labels, save_path):
    """Plot conditions × sublayers heatmap."""
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.8), 4))
    sns.heatmap(
        matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        cbar_kws={"label": "Attention weight on most recent value"},
    )
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Cardiac Condition")
    ax.set_title("Depth-wise Attention Patterns by Diagnostic Condition")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {save_path}")


def plot_depth_distributions(condition_alphas, save_path):
    """Line plot: for each condition, show how attention is distributed
    across depth at a selected sublayer (e.g., the last layer's attention sublayer).

    This shows the full alpha distribution, not just the last value.
    """
    fig, axes = plt.subplots(1, len(condition_alphas), figsize=(4 * len(condition_alphas), 4), sharey=True)

    if len(condition_alphas) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, SUPERCLASSES):
        if class_name not in condition_alphas:
            continue

        alphas = condition_alphas[class_name]
        # Pick the last attention sublayer (second to last entry, before final)
        last_attn_alpha = alphas[-3] if len(alphas) >= 3 else alphas[-1]

        ax.bar(range(len(last_attn_alpha)), last_attn_alpha, color="steelblue")
        ax.set_title(class_name)
        ax.set_xlabel("Value index (depth)")
        if ax == axes[0]:
            ax.set_ylabel("Attention weight")

    plt.suptitle("Depth Attention Distribution (Last Attention Sublayer)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved depth distributions to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/ptbxl")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.config, args.checkpoint, device)

    # Load test data (fold 10)
    # TODO: load PTB-XL test split using utils.data functions
    df = load_metadata(args.data_dir)
    signals = load_signals(df, args.data_dir, sampling_rate=100)
    splits = get_splits(df, signals)
    X_test, y_test = splits["X_test"], splits["y_test"]

    # TODO: create DataLoader with batch_size, shuffle=False
    test_dataset = PTBXLDataset(X_test, y_test, augment=False)
    from torch.utils.data import DataLoader 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # raise NotImplementedError("Wire up data loading here")

    # Extract attention weights
    print("Extracting attention weights from test set...")
    all_alphas, all_labels, all_preds = extract_alphas(model, test_loader, device)
    print(f"Collected alphas for {len(all_alphas)} samples")

    # Aggregate by condition
    condition_alphas, sample_counts = aggregate_by_condition(all_alphas, all_labels)
    for name, count in sample_counts.items():
        print(f"  {name}: {count} samples")

    # Build and plot heatmap
    matrix, row_labels, col_labels = build_heatmap_matrix(condition_alphas)
    plot_heatmap(matrix, row_labels, col_labels, output_dir / "attention_heatmap.png")

    # Plot depth distributions
    plot_depth_distributions(condition_alphas, output_dir / "depth_distributions.png")

    # Save raw data for further analysis
    np.savez(
        output_dir / "attention_weights_raw.npz",
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        sample_counts=sample_counts,
    )
    print(f"Saved raw data to {output_dir / 'attention_weights_raw.npz'}")


if __name__ == "__main__":
    main()