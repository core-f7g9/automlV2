# multi-target-pipeline/preprocess.py
# ============================================================
# Cell 2: Write processing script — independent per-target splits
# ============================================================
import textwrap, os

split_script = textwrap.dedent("""
import argparse, os, glob
import pandas as pd
import numpy as np

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

def per_target_split(df, target_col, input_feats, val_frac=0.2, seed=42, min_support=5, rare_train_only=True):
    df_t = df[~df[target_col].isna()].copy()
    df_t[target_col] = df_t[target_col].astype(str)

    counts = df_t[target_col].value_counts(dropna=False)
    rng = np.random.RandomState(seed)

    train_idx = []
    val_idx = []

    for cls, g in df_t.groupby(target_col):
        n = len(g)
        if n < min_support and rare_train_only:
            train_idx.extend(g.index.tolist())
            continue

        n_val = int(round(n * val_frac))
        n_val = min(n_val, n - 1) if n > 1 else 0
        if n_val <= 0:
            train_idx.extend(g.index.tolist())
        else:
            val_take = rng.choice(g.index.values, size=n_val, replace=False)
            val_idx.extend(val_take.tolist())
            train_idx.extend(sorted(set(g.index.values) - set(val_take)))

    train = df_t.loc[sorted(train_idx)]
    val   = df_t.loc[sorted(val_idx)]

    keep_cols = list(input_feats) + [target_col]
    train = train[keep_cols]
    val   = val[keep_cols]

    if train[target_col].nunique(dropna=True) < 2 and counts.nunique() > 1:
        moved = False
        val_groups = val.groupby(target_col)
        for cls, g in val_groups:
            if cls not in set(train[target_col].unique()):
                idx = g.index[0]
                train = pd.concat([train, val.loc[[idx]]], axis=0)
                val = val.drop(index=[idx])
                moved = True
                if train[target_col].nunique(dropna=True) >= 2:
                    break

    return train, val, counts.to_dict()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets_csv", type=str, required=True)
    p.add_argument("--input_features_csv", type=str, required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--min_support", type=int, default=5)
    p.add_argument("--rare_train_only", type=str, default="true")
    p.add_argument("--mounted_input_dir", type=str, default="/opt/ml/processing/input")
    p.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    args = p.parse_args()

    rare_train_only = str(args.rare_train_only).lower() in ("1","true","yes","y","t")

    local_in = find_input_csv(args.mounted_input_dir)
    df = pd.read_csv(local_in, low_memory=False)

    targets = [c.strip() for c in args.targets_csv.split(",") if c.strip()]
    input_feats = [c.strip() for c in args.input_features_csv.split(",") if c.strip()]

    for c in input_feats + targets:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV header")

    for tgt in targets:
        train, val, counts = per_target_split(
            df=df,
            target_col=tgt,
            input_feats=input_feats,
            val_frac=args.val_frac,
            seed=args.random_seed,
            min_support=args.min_support,
            rare_train_only=rare_train_only
        )

        out_dir_tr = os.path.join(args.output_dir, tgt, "train")
        out_dir_va = os.path.join(args.output_dir, tgt, "validation")
        os.makedirs(out_dir_tr, exist_ok=True)
        os.makedirs(out_dir_va, exist_ok=True)

        train.to_csv(os.path.join(out_dir_tr, "train.csv"), index=False)
        val.to_csv(os.path.join(out_dir_va, "validation.csv"), index=False)

if __name__ == "__main__":
    main()
""").strip()

with open("prepare_per_target_splits.py", "w") as f:
    f.write(split_script)

print("Wrote prepare_per_target_splits.py")
