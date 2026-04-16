"""Optional: preprocess and cache PTB-XL signals as a single .npz for faster loading."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/ptbxl_cached.npz")
    parser.add_argument("--sampling_rate", type=int, default=100)
    args = parser.parse_args()

    # TODO: load all signals via utils.data.load_signals
    # TODO: z-score per lead across training set, save means/stds
    # TODO: save to .npz: X, y_superclass, y_subclass, strat_fold, patient_id

    raise NotImplementedError


if __name__ == "__main__":
    main()
