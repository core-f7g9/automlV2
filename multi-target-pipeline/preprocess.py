# ============================================================
# Cell 2: Write processing script — split once, write per-target
# ============================================================
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
    p.add_argument("--stratify_target", type=str, required=True)
    p.add_argument("--targets_csv", type=str, required=True, help="Comma-separated target columns to produce")
    p.add_argument("--input_features_csv", type=str, required=True, help="Comma-separated feature columns to keep")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--mounted_input_dir", type=str, default="/opt/ml/processing/input")
    p.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    args = p.parse_args()

    local_in = find_input_csv(args.mounted_input_dir)
    df = pd.read_csv(local_in, low_memory=False)

    targets = [c.strip() for c in args.targets_csv.split(",") if c.strip()]
    input_feats = [c.strip() for c in args.input_features_csv.split(",") if c.strip()]

    # Basic validations
    for c in input_feats + targets:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV header")
    if args.stratify_target not in targets:
        raise ValueError("--stratify_target must be one of the targets being produced")

    # Drop rows missing the stratify target
    before = len(df)
    df = df[~df[args.stratify_target].isna()].copy()
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing stratify target '{args.stratify_target}'.")

    # Ensure ≥2 classes for the stratify target
    if df[args.stratify_target].nunique(dropna=True) < 2:
        raise ValueError(f"Need at least 2 classes in stratify target '{args.stratify_target}'.")

    # Split once (on stratify target)
    train_df, val_df = stratified_split(df, args.stratify_target, args.val_frac, args.random_seed)

    # For each target, write INPUT_FEATURES + that target, into separate folders
    for tgt in targets:
        tr = train_df[~train_df[tgt].isna()].copy()
        va = val_df[~val_df[tgt].isna()].copy()

        keep_cols = input_feats + [tgt]
        tr = tr[keep_cols]
        va = va[keep_cols]

        # Sanity: both splits must have at least 2 classes for tgt
        if tr[tgt].nunique(dropna=True) < 2:
            raise ValueError(f"Target '{tgt}' has <2 classes in train after filtering.")
        combined = pd.concat([tr[[tgt]], va[[tgt]]], axis=0)
        missing_in_train = set(combined[tgt].unique()) - set(tr[tgt].unique())
        if missing_in_train:
            raise ValueError(f"Target '{tgt}': some classes absent in train after split: {missing_in_train}")

        out_dir_tr = os.path.join(args.output_dir, tgt, "train")
        out_dir_va = os.path.join(args.output_dir, tgt, "validation")
        os.makedirs(out_dir_tr, exist_ok=True)
        os.makedirs(out_dir_va, exist_ok=True)

        tr.to_csv(os.path.join(out_dir_tr, "train.csv"), index=False)
        va.to_csv(os.path.join(out_dir_va, "validation.csv"), index=False)
        print(f"[{tgt}] wrote train={len(tr)} validation={len(va)} with columns={keep_cols}")

if __name__ == "__main__":
    main()
""").strip()

with open("prepare_per_target_splits.py", "w") as f:
    f.write(script)

print("Wrote prepare_per_target_splits.py")
