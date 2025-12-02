
import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

from feature_encoder import FeatureEncoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cat_cols", nargs="*", default=["VendorName"])
    p.add_argument("--text_cols", nargs="*", default=["LineDescription"])
    p.add_argument("--num_cols", nargs="*", default=["ClubNumber"])
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=[args.target])

    df = df.sample(frac=1, random_state=42)
    n_val = max(int(len(df) * 0.2), 1)
    if n_val >= len(df):
        n_val = max(len(df) - 1, 0)

    df_train = df.iloc[:-n_val] if n_val else df
    df_val = df.iloc[-n_val:] if n_val else df.iloc[0:0]

    fe = FeatureEncoder(
        cat_cols=args.cat_cols,
        text_cols=args.text_cols,
        num_cols=args.num_cols
    )

    X_train = fe.fit_transform(df_train)
    X_val = fe.transform(df_val)

    target_le = LabelEncoder()
    y_train = target_le.fit_transform(df_train[args.target].astype(str))
    y_val = target_le.transform(df_val[args.target].astype(str))

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(fe, os.path.join(args.output_dir, "fe.pkl"))
    joblib.dump(target_le, os.path.join(args.output_dir, "label_encoder.pkl"))

    train_out = pd.DataFrame(X_train)
    train_out["target"] = y_train

    val_out = pd.DataFrame(X_val)
    val_out["target"] = y_val

    train_out.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_out.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)


if __name__ == "__main__":
    main()
