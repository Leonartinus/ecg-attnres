# ecg-attnres

Attention Residuals for 12-Lead ECG Cardiac Diagnosis on PTB-XL.

## Quick start (Google Colab)

```python
!git clone https://github.com/<YOUR_USERNAME>/ecg-attnres.git
%cd ecg-attnres
!pip install -r requirements.txt

# Mount Drive so PTB-XL and checkpoints persist across sessions
from google.colab import drive
drive.mount('/content/drive')

# One-time data download (only first session)
!python scripts/download_ptbxl.py --dest /content/drive/MyDrive/ecg-attnres/data

# Run an experiment
!python train.py --config configs/baseline_prenorm.yaml --seed 42
```

## Repo layout

```
models/      model components (tokenizer, transformer, AttnRes, classifier)
utils/       data loading, metrics, visualization
configs/     YAML configs, one per experiment
scripts/     one-off helpers (data download, preprocessing)
notebooks/   thin entry points (explore data, run experiments, analyze)
tests/       unit tests (quick sanity checks before long runs)
analysis/    standalone analysis scripts (attention heatmaps, lead ablation)
data/        PTB-XL (not committed - use Drive in Colab)
checkpoints/ saved models (not committed)
logs/        training logs (not committed)
outputs/     result CSVs (not committed)
figures/     generated figures
```

## Development workflow

1. Edit `.py` files locally (VS Code) or in the Colab file editor.
2. Commit and push to GitHub frequently — each person on a branch.
3. Merge via PR. `.py` files diff cleanly; notebooks do not.
4. Run experiments from notebooks via `!python train.py --config ...`.

## Team conventions

- **Never commit notebooks with outputs.** Run `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` before committing.
- **Never commit data files or checkpoints.** They live in Drive.
- **One config per experiment.** Config name = checkpoint name = log name.
- **Seeds**: use 5 seeds per model — 42, 43, 44, 45, 46.
