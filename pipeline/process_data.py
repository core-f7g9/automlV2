# ============================================
# Cell 2: Write the processing script to disk
# ============================================
# This script reads CSV from S3, does a stratified split, writes train/validation.
script = textwrap.dedent("""
import argparse, os
import pandas as pd
import numpy as np
import awswrangler as wr

def stratified_split(df, target_col, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    val_idx = []
    for cls, g in df.groupby(target_col):
        n = max(1, int(len(g) * val_frac))
        val_idx.extend(rng.choice(g.index.values, size=n, replace=False))
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

    df = wr.s3.read_csv(args.input_s3_csv)
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not in columns: {df.columns.tolist()}")

    train, val = stratified_split(df, args.target_col, args.val_frac, args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.output_dir, "validation.csv"), index=False)

if __name__ == "__main__":
    main()
""").strip()

with open("sql_to_s3_and_split.py", "w") as f:
    f.write(script)

print("Wrote sql_to_s3_and_split.py")