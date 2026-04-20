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
import ast


# Superclass mapping used for the primary 5-class task
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load ptbxl_database.csv, parse scp_codes column, return DataFrame."""
    # TODO: read ptbxl_database.csv
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    # TODO: parse scp_codes (stored as literal dict strings)
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    # TODO: merge with scp_statements.csv to get superclass mapping
    scp_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    df = aggregate_superclass(df, scp_df)

    return df


def aggregate_superclass(df: pd.DataFrame, scp_df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot-like columns for each of the 5 diagnostic superclasses."""
    # TODO: for each scp_code present in the record, look up its superclass
    # TODO: create binary columns NORM, MI, STTC, CD, HYP
    agg_df = scp_df[scp_df.diagnostic == 1]
    def aggregate_diagnostic(y_dic):
        tmp = [0, 0, 0, 0, 0]
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp[SUPERCLASSES.index(agg_df.loc[key].diagnostic_class)] = 1
        return tmp
    df[SUPERCLASSES] = df.scp_codes.apply(aggregate_diagnostic).apply(pd.Series)

    return df


def load_signals(df: pd.DataFrame, data_dir: Path, sampling_rate: int = 100) -> np.ndarray:
    """Load raw ECG signals via wfdb. Returns array of shape (N, timesteps, 12)."""
    import wfdb
    # TODO: iterate over df.filename_lr (100Hz) or df.filename_hr (500Hz)
    # TODO: wfdb.rdsamp each file; stack
    if sampling_rate == 100:
        data = [wfdb.rdsamp(data_dir / f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(data_dir / f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])

    return data


def get_splits(df: pd.DataFrame, X: np.ndarray) -> dict:
    """Split by recommended strat_fold: 1-8 train, 9 val, 10 test."""
    # TODO: return {'X_train', 'y_train', 'X_val', ...} using df.strat_fold
    df_train = df[df.strat_fold <= 8]
    df_val = df[df.strat_fold == 9]
    df_test = df[df.strat_fold == 10]

    return {
        "X_train": X[df_train.index],
        "y_train": df_train[SUPERCLASSES].values,
        "X_val": X[df_val.index],
        "y_val": df_val[SUPERCLASSES].values,
        "X_test": X[df_test.index],
        "y_test": df_test[SUPERCLASSES].values
    }


def compute_pos_weights(y_train: np.ndarray) -> torch.Tensor:
    """BCE pos_weight per class: (n_negatives / n_positives), clipped to sensible range."""
    # TODO: handle zero-positive edge case
    pos_weights = []
    for i in range(y_train.shape[1]):
        n_positives = y_train[:, i].sum()
        n_negatives = y_train.shape[0] - n_positives
        if n_positives == 0:
            pos_weights.append(1.0)
        else:
            pos_weights.append(n_negatives / n_positives)
    return torch.tensor(pos_weights)


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
        if self.augment:
            # Random time shift
            shift = np.random.randint(-50, 51)
            self.X[idx] = np.roll(self.X[idx], shift, axis=0)
            # Random amplitude scaling
            scale = np.random.uniform(0.9, 1.1)
            self.X[idx] = self.X[idx] * scale
        
        # TODO: z-score normalize per lead
        self.X[idx] = (self.X[idx] - self.X[idx].mean()) / (self.X[idx].std() + 1e-8)

        # TODO: return (signal_tensor, label_tensor) [+ demographics if provided]
        signal = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.demographics is not None:
            demographics = torch.tensor(self.demographics[idx], dtype=torch.float32)
            return signal, label, demographics
        
        return signal, label
