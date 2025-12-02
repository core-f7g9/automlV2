import textwrap

preprocess_code = textwrap.dedent("""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def build_fe_pipeline(cat_cols, text_cols, num_cols):

    transformers = []

    if cat_cols:
        transformers.append(("cat_label", LabelEncoderTransformer(cat_cols), cat_cols))

    if text_cols:
        transformers.append(("text_tfidf", TfidfVectorizerTransformer(text_cols), text_cols))

    if num_cols:
        transformers.append(("num_scale", StandardScaler(), num_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ("vendor_label", LabelEncoderTransformer(["VendorName"]), ["VendorName"]),
            ("line_tfidf", TfidfVectorizerTransformer(["LineDescription"]), "LineDescription"),
            ("num_scale", StandardScaler(), ["ClubNumber"])
        ],
        remainder="drop"
    )

    return preprocessor


class LabelEncoderTransformer:
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str).fillna("Unknown")
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        out = []
        for col in self.columns:
            out.append(self.encoders[col].transform(X[col].astype(str).fillna("Unknown")))
        return np.array(out).T


class TfidfVectorizerTransformer:
    def __init__(self, columns):
        self.columns = columns
        self.vectorizers = {}

    def fit(self, X, y=None):
        for col in self.columns:
            tf = TfidfVectorizer(max_features=5000)
            tf.fit(X[col].astype(str).fillna(""))
            self.vectorizers[col] = tf
        return self

    def transform(self, X):
        out = []
        for col in self.columns:
            vec = self.vectorizers[col].transform(X[col].astype(str).fillna("")).toarray()
            out.append(vec)
        return np.hstack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)

    # Drop missing target rows
    df = df.dropna(subset=[args.target])

    # Split train/val
    df = df.sample(frac=1, random_state=42)
    n_val = int(len(df) * 0.2)
    df_train = df.iloc[:-n_val]
    df_val   = df.iloc[-n_val:]

    df_train["ClubNumber"] = pd.to_numeric(df_train["ClubNumber"], errors="coerce").fillna(0)
    df_val["ClubNumber"]   = pd.to_numeric(df_val["ClubNumber"], errors="coerce").fillna(0)                                      

    # Build FE pipeline
    fe = build_fe_pipeline(
        cat_cols=["VendorName"],
        text_cols=["LineDescription"],
        num_cols=["ClubNumber"]
    )

    # Fit FE
    X_train = fe.fit_transform(df_train)
    X_val   = fe.transform(df_val)

    # Save FE pipeline
    joblib.dump(fe, os.path.join(args.output_dir, "fe.pkl"))

    # Save transformed datasets
    train_out = pd.DataFrame(X_train)
    train_out["target"] = df_train[args.target].values

    val_out = pd.DataFrame(X_val)
    val_out["target"] = df_val[args.target].values

    train_out.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_out.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)

if __name__ == "__main__":
    main()
""")

with open("preprocess_fe.py", "w") as f:
    f.write(preprocess_code)

print("Wrote preprocess_fe.py")
