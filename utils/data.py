"""PTB-XL data loading and preprocessing.

Expected directory layout after download:
    data/ptbxl/
        ptbxl_database.csv
        scp_statements.csv
        records100/       (100 Hz signals)
        records500/       (500 Hz signals)
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Superclass mapping used for the primary 5-class task
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load ptbxl_database.csv, parse scp_codes column, return DataFrame."""
    # TODO: read ptbxl_database.csv
    # TODO: parse scp_codes (stored as literal dict strings)
    # TODO: merge with scp_statements.csv to get superclass mapping
    raise NotImplementedError


def aggregate_superclass(df: pd.DataFrame, scp_df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot-like columns for each of the 5 diagnostic superclasses."""
    # TODO: for each scp_code present in the record, look up its superclass
    # TODO: create binary columns NORM, MI, STTC, CD, HYP
    raise NotImplementedError


def load_signals(df: pd.DataFrame, data_dir: Path, sampling_rate: int = 100) -> np.ndarray:
    """Load raw ECG signals via wfdb. Returns array of shape (N, timesteps, 12)."""
    import wfdb
    # TODO: iterate over df.filename_lr (100Hz) or df.filename_hr (500Hz)
    # TODO: wfdb.rdsamp each file; stack
    raise NotImplementedError


def get_splits(df: pd.DataFrame, X: np.ndarray) -> dict:
    """Split by recommended strat_fold: 1-8 train, 9 val, 10 test."""
    # TODO: return {'X_train', 'y_train', 'X_val', ...} using df.strat_fold
    raise NotImplementedError


def compute_pos_weights(y_train: np.ndarray) -> torch.Tensor:
    """BCE pos_weight per class: (n_negatives / n_positives), clipped to sensible range."""
    # TODO: handle zero-positive edge case
    raise NotImplementedError


class PTBXLDataset(Dataset):
    """PyTorch Dataset wrapping preloaded signals and labels.

    Augmentations (optional, applied only on training split):
        - random time shift +/- 50 samples
        - random amplitude scaling 0.9-1.1
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, demographics: np.ndarray = None,
                 augment: bool = False):
        self.X = X
        self.y = y
        self.demographics = demographics
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: apply augmentation if self.augment
        # TODO: z-score normalize per lead
        # TODO: return (signal_tensor, label_tensor) [+ demographics if provided]
        raise NotImplementedError
