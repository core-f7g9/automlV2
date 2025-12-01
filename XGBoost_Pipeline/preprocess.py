# ============================================================
# Cell 2: Write processing script — XGBoost compatible
# ============================================================
import textwrap

split_script = textwrap.dedent("""
import argparse, os, glob, json
import pandas as pd
import numpy as np

def find_input_csv(path):
    c = glob.glob(os.path.join(path, '*.csv'))
    if c: return c[0]
    raise FileNotFoundError("No CSV found in mounted input path")

def per_target_split(df, tgt, feats, val_frac=0.2, seed=42, min_support=5, rare_train_only=True):
    df = df.dropna(subset=[tgt]).copy()
    df[tgt] = df[tgt].astype(str)

    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []

    for cls, g in df.groupby(tgt):
        n = len(g)
        if n < min_support and rare_train_only:
            train_idx.extend(g.index)
            continue

        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        if n_val == 0:
            train_idx.extend(g.index)
        else:
            val_take = rng.choice(g.index, size=n_val, replace=False)
            val_idx.extend(val_take)
            train_idx.extend(list(set(g.index) - set(val_take)))

    train = df.loc[train_idx]
    val   = df.loc[val_idx]

    # Ensure target appears in training
    for cls in val[tgt].unique():
        if cls not in train[tgt].unique():
            idx = val[val[tgt] == cls].index[0]
            train = pd.concat([train, val.loc[[idx]]])
            val = val.drop(index=[idx])

    # Label encode
    classes = sorted(train[tgt].unique())
    cls_map = {c: i for i, c in enumerate(classes)}
    train[tgt] = train[tgt].map(cls_map)
    val[tgt]   = val[tgt].map(cls_map)

    # XGBoost format – label in col 0, NO header
    cols = [tgt] + feats
    train = train[cols]
    val   = val[cols]

    return train, val, len(classes)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets_csv", type=str)
    p.add_argument("--input_features_csv", type=str)
    p.add_argument("--val_frac", type=float)
    p.add_argument("--random_seed", type=int)
    p.add_argument("--min_support", type=int)
    p.add_argument("--rare_train_only", type=str)
    p.add_argument("--mounted_input_dir", type=str)
    p.add_argument("--output_dir", type=str)
    args = p.parse_args()

    rare = args.rare_train_only.lower() in ("true","1","yes")
    df = pd.read_csv(find_input_csv(args.mounted_input_dir))

    targets = args.targets_csv.split(",")
    feats   = args.input_features_csv.split(",")

    for t in targets:
        tr, va, num_classes = per_target_split(
            df, t, feats,
            val_frac=args.val_frac,
            seed=args.random_seed,
            min_support=args.min_support,
            rare_train_only=rare
        )

        out_tr = os.path.join(args.output_dir, t, "train")
        out_va = os.path.join(args.output_dir, t, "validation")
        os.makedirs(out_tr, exist_ok=True)
        os.makedirs(out_va, exist_ok=True)

        tr.to_csv(os.path.join(out_tr, "train.csv"), index=False, header=False)
        va.to_csv(os.path.join(out_va, "validation.csv"), index=False, header=False)

        with open(os.path.join(out_tr, "classes.json"), "w") as f:
            json.dump({"num_class": num_classes}, f)

if __name__ == "__main__":
    main()
""")

with open("prepare_per_target_splits.py", "w") as f:
    f.write(split_script)

print("Wrote prepare_per_target_splits.py")
