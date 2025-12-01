# ============================================================
# Cell 2: Write processing script — XGBoost compatible
# ============================================================
import textwrap

script = textwrap.dedent("""
import argparse, os, glob, json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# -----------------
# Text -> vector
# -----------------
def vectorize_text(df, col, n_features=64):
    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None
    )
    # Replace nulls with empty string
    texts = df[col].fillna("").astype(str).tolist()
    mat = vec.transform(texts).toarray()

    new_cols = [f"{col}_{i}" for i in range(n_features)]
    df[new_cols] = mat
    df.drop(columns=[col], inplace=True)
    return df


# -----------------
# Find CSV
# -----------------
def find_input_csv(path):
    candidates = glob.glob(os.path.join(path, "*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No CSV found in mounted input directory")


# -----------------
# Core split logic
# -----------------
def per_target_split(df, tgt, feats, val_frac, seed, min_support, rare_train_only):

    df = df.dropna(subset=[tgt]).copy()
    df[tgt] = df[tgt].astype(str)

    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []

    # Class-wise stratification
    for cls, g in df.groupby(tgt):
        n = len(g)
        if n < min_support and rare_train_only:
            train_idx.extend(g.index.tolist())
            continue

        n_val = int(round(n * val_frac)) if n > 1 else 0
        n_val = min(n_val, n - 1)
        if n_val <= 0:
            train_idx.extend(g.index.tolist())
        else:
            chosen = rng.choice(g.index.values, size=n_val, replace=False)
            val_idx.extend(chosen.tolist())
            train_idx.extend(sorted(set(g.index) - set(chosen)))

    train = df.loc[train_idx].copy()
    val   = df.loc[val_idx].copy()

    # Ensure all classes appear in train
    for cls in val[tgt].unique():
        if cls not in train[tgt].unique():
            idx = val[val[tgt] == cls].index[0]
            train = pd.concat([train, val.loc[[idx]]])
            val   = val.drop(index=[idx])

    # ----------------------------------
    # LABEL ENCODE
    # ----------------------------------
    classes = sorted(train[tgt].unique())
    clsmap = {c: i for i, c in enumerate(classes)}
    train[tgt] = train[tgt].map(clsmap)
    val[tgt]   = val[tgt].map(clsmap)

    # ----------------------------------
    # TEXT → NUMERIC VECTOR (64 dims)
    # ----------------------------------
    for col in feats:
        if train[col].dtype == object or val[col].dtype == object:
            train = vectorize_text(train, col, n_features=64)
            val   = vectorize_text(val, col, n_features=64)

    # Order: label first, then numeric features
    feature_cols = [c for c in train.columns if c != tgt]
    cols = [tgt] + feature_cols

    return train[cols], val[cols], len(classes)


# -----------------
# Main entry
# -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets_csv")
    p.add_argument("--input_features_csv")
    p.add_argument("--val_frac", type=float)
    p.add_argument("--random_seed", type=int)
    p.add_argument("--min_support", type=int)
    p.add_argument("--rare_train_only")
    p.add_argument("--mounted_input_dir")
    p.add_argument("--output_dir")
    args = p.parse_args()

    rare = args.rare_train_only.lower() in ("true","1","yes")

    df = pd.read_csv(find_input_csv(args.mounted_input_dir))

    targets = args.targets_csv.split(",")
    feats   = args.input_features_csv.split(",")

    for tgt in targets:
        train, val, nclass = per_target_split(
            df, tgt, feats,
            args.val_frac,
            args.random_seed,
            args.min_support,
            rare
        )

        out_tr = os.path.join(args.output_dir, tgt, "train")
        out_va = os.path.join(args.output_dir, tgt, "validation")

        os.makedirs(out_tr, exist_ok=True)
        os.makedirs(out_va, exist_ok=True)

        # XGBoost CSV rules → NO HEADER
        train.to_csv(os.path.join(out_tr, "train.csv"), index=False, header=False)
        val.to_csv(os.path.join(out_va, "validation.csv"), index=False, header=False)

        # Save num_class
        with open(os.path.join(out_tr, "classes.json"), "w") as f:
            json.dump({"num_class": nclass}, f)


if __name__ == "__main__":
    main()
""")

with open("prepare_per_target_splits.py", "w") as f:
    f.write(script)

print("Wrote fixed XGBoost-compatible preprocess script")

