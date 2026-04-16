"""One-time PTB-XL download from PhysioNet.

Usage:
    python scripts/download_ptbxl.py --dest data/ptbxl
    # or in Colab with Drive mounted:
    python scripts/download_ptbxl.py --dest /content/drive/MyDrive/ecg-attnres/data/ptbxl
"""
import argparse
import subprocess
from pathlib import Path


PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, required=True)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Use wget to mirror the PTB-XL files (~2.3 GB)
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--cut-dirs=4", "-nH",
        PTBXL_URL,
        "-P", str(dest),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Downloaded PTB-XL to {dest}")


if __name__ == "__main__":
    main()
