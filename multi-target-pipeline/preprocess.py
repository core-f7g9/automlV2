# ============================================
# Cell 2: Write the processing script to disk
# ============================================
# Reads CSV from the mounted ProcessingInput, does a stratified split, writes train/validation.
import textwrap, os

script = textwrap.dedent("""
import argparse, os, glob
import pandas as pd
import numpy as np

def stratified_split(df, target_col, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    val_idx = []
    for cls, g in df.groupby(target_col):
        if len(g) == 1:
            continue
        n_val = int(len(g) * val_frac)
        n_val = max(1, n_val)
        n_val = min(n_val, len(g) - 1)
        val_idx.extend(rng.choice(g.index.values, size=n_val, replace=False))
    val = df.loc[val_idx]
    train = df.drop(index=val_idx)
    return train, val

def find_input_csv(mounted_dir: str) -> str:
    candidates = glob.glob(os.path.join(mounted_dir, "*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    fallback = os.path.join(mounted_dir, "data.csv")
    if os.path.exists(fallback):
        return fallback
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(f"No CSV found under {mounted_dir}. Ensure your S3 object ends with .csv")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--mounted_input_dir", type=str, default="/opt/ml/processing/input")
    p.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    args = p.parse_args()

    local_in = find_input_csv(args.mounted_input_dir)
    df = pd.read_csv(local_in, low_memory=False)

    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not in columns: {df.columns.tolist()}")

    before = len(df)
    df = df[~df[args.target_col].isna()].copy()
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing target.")

    n_classes = df[args.target_col].nunique(dropna=True)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes in target '{args.target_col}', found {n_classes}.")

    print("Class counts after target-drop:")
    print(df[args.target_col].value_counts(dropna=False))

    train, val = stratified_split(df, args.target_col, args.val_frac, args.random_seed)

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

print("Wrote sql_to_s3_and_split.py")
