# ============================
# Cell 2 — Write preprocess script
# ============================

import textwrap

preprocess_script = textwrap.dedent("""
import argparse, os, glob, json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

# ----------------------------
# Utility
# ----------------------------

def find_csv(mounted_dir):
    files = glob.glob(os.path.join(mounted_dir, "*.csv"))
    if files:
        return files[0]
    raise FileNotFoundError("No CSV found in mounted input")

# Rare-class aware splitting (same as your S1)
def per_target_split(df, target_col, input_feats,
                     val_frac=0.2, seed=42,
                     min_support=5, rare_train_only=True):

    df_t = df.dropna(subset=[target_col]).copy()
    df_t[target_col] = df_t[target_col].astype(str)

    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []

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

    keep = input_feats + [target_col]
    return train[keep], val[keep]

# ----------------------------
# Hybrid TF-IDF
# ----------------------------

def build_hybrid_tfidf(train_df, val_df, text_cols):
    # Convert numeric "text-like" fields (e.g., ClubNumber)
    for c in text_cols:
        train_df[c] = train_df[c].astype(str)
        val_df[c]   = val_df[c].astype(str)

    # Combine text columns into one field
    def combine(df):
        return df[text_cols].fillna("").agg(" ".join, axis=1)

    train_text = combine(train_df)
    val_text   = combine(val_df)

    # Hybrid vectorizer
    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        max_features=20000,
        min_df=2
    )

    vec_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        max_features=30000,
        min_df=2
    )

    Xw = vec_word.fit_transform(train_text)
    Xc = vec_char.fit_transform(train_text)

    X_train = sparse.hstack([Xw, Xc]).tocsr()
    X_val   = sparse.hstack([
        vec_word.transform(val_text),
        vec_char.transform(val_text)
    ]).tocsr()

    return X_train, X_val, vec_word, vec_char

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets_csv")
    p.add_argument("--inputs_csv")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_support", type=int, default=5)
    p.add_argument("--rare_train_only", type=str, default="true")
    p.add_argument("--input_dir", default="/opt/ml/processing/input")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    args = p.parse_args()

    rare = args.rare_train_only.lower() in ("true","1","y")

    df = pd.read_csv(find_csv(args.input_dir), low_memory=False)
    targets = [x.strip() for x in args.targets_csv.split(",")]
    feats   = [x.strip() for x in args.inputs_csv.split(",")]

    for tgt in targets:
        train_df, val_df = per_target_split(
            df, tgt, feats,
            args.val_frac, args.seed,
            args.min_support, rare
        )

        # Hybrid TF-IDF
        X_train, X_val, vec_word, vec_char = build_hybrid_tfidf(
            train_df.copy(),
            val_df.copy(),
            feats
        )

        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(train_df[tgt])
        y_val   = le.transform(val_df[tgt])

        out = os.path.join(args.output_dir, tgt)
        os.makedirs(out, exist_ok=True)

        sparse.save_npz(os.path.join(out, "X_train.npz"), X_train)
        sparse.save_npz(os.path.join(out, "X_val.npz"),   X_val)
        np.save(os.path.join(out, "y_train.npy"), y_train)
        np.save(os.path.join(out, "y_val.npy"),   y_val)

        # Save preprocess artifacts
        import joblib
        joblib.dump(vec_word, os.path.join(out, "tfidf_word.pkl"))
        joblib.dump(vec_char, os.path.join(out, "tfidf_char.pkl"))
        joblib.dump(le,        os.path.join(out, "label_encoder.pkl"))

        # Save config
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"text_cols": feats, "target": tgt}, f)

if __name__ == "__main__":
    main()
""")

with open("preprocess_hybrid.py", "w") as f:
    f.write(preprocess_script)

print("Wrote preprocess_hybrid.py")
