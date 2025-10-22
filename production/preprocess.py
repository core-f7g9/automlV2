# ============================================
# Cell 2: Write the processing script to disk
# ============================================
# This script reads CSV from S3, does a stratified split, writes train/validation.
import textwrap, os

script = textwrap.dedent("""
import argparse, os
import pandas as pd
import numpy as np
import boto3
from urllib.parse import urlparse

def parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {uri}")
    return bucket, key

def download_s3_to_local(s3_uri: str, local_path: str):
    bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path

def stratified_split(df, target_col, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    val_idx = []
    for cls, g in df.groupby(target_col):
        # Ensure at least 1 sample goes to validation AND at least 1 stays for training
        n_val = int(len(g) * val_frac)
        n_val = max(1, n_val)
        n_val = min(n_val, len(g) - 1)  # keep at least 1 for training
        # If a class has only 1 row, keep it in training
        if len(g) == 1:
            continue
        val_idx.extend(rng.choice(g.index.values, size=n_val, replace=False))
    val = df.loc[val_idx]
    train = df.drop(index=val_idx)
    return train, val

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_s3_csv", type=str, required=True)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    args = p.parse_args()

    # Download CSV from S3 to local path (no awswrangler needed)
    local_in = "/opt/ml/processing/input/input.csv"
    download_s3_to_local(args.input_s3_csv, local_in)

    # Read CSV (large-file friendly)
    df = pd.read_csv(local_in, low_memory=False)

    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not in columns: {df.columns.tolist()}")

    # Drop rows with missing target
    before = len(df)
    df = df[~df[args.target_col].isna()].copy()
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing target.")

    # Must have at least 2 classes
    n_classes = df[args.target_col].nunique(dropna=True)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes in target '{args.target_col}', found {n_classes}.")

    train, val = stratified_split(df, args.target_col, args.val_frac, args.random_seed)

    # Final sanity: every class present in train
    missing_in_train = set(df[args.target_col].unique()) - set(train[args.target_col].unique())
    if missing_in_train:
        raise ValueError(f"Some classes have no training examples after split: {missing_in_train}. "
                         f"Consider lowering --val_frac or reviewing class counts.")

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "validation"), exist_ok=True)

    train.to_csv(os.path.join(args.output_dir, "train", "train.csv"), index=False)
    val.to_csv(os.path.join(args.output_dir, "validation", "validation.csv"), index=False)
    print(f"Wrote train={len(train)} rows, validation={len(val)} rows.")

if __name__ == "__main__":
    main()
""").strip()

with open("sql_to_s3_and_split.py", "w") as f:
    f.write(script)

print("Wrote sql_to_s3_and_split.py (no awswrangler needed, robust split)")