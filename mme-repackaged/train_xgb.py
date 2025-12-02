
import argparse
import json
import os
import shutil

import boto3
import joblib
import pandas as pd
import xgboost as xgb


def resolve_path(cli_value, env_key, filename=None):
    if cli_value:
        return cli_value
    base = os.environ.get(env_key)
    if not base:
        raise ValueError(f"Missing required path for {env_key}")
    return os.path.join(base, filename) if filename else base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv")
    p.add_argument("--val_csv")
    p.add_argument("--fe_model")
    p.add_argument("--label_encoder")
    p.add_argument("--output_dir")
    p.add_argument("--hp")
    args = p.parse_args()

    train_csv = resolve_path(args.train_csv, "SM_CHANNEL_TRAIN", "train.csv")
    val_csv = resolve_path(args.val_csv, "SM_CHANNEL_VALIDATION", "val.csv")
    fe_path = resolve_path(args.fe_model, "SM_CHANNEL_FE_MODEL", "fe.pkl")
    label_encoder_path = resolve_path(args.label_encoder, "SM_CHANNEL_LABEL_ENCODER", "label_encoder.pkl")
    output_dir = resolve_path(args.output_dir, "SM_MODEL_DIR")
    hp_path = args.hp

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    y_train = df_train["target"].values
    y_val = df_val["target"].values

    X_train = df_train.drop(columns=["target"])
    X_val = df_val.drop(columns=["target"])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val) if len(X_val) else None

    params = {
        "objective": "multi:softprob",
        "num_class": len(pd.unique(y_train)),
        "max_depth": 6,
        "eta": 0.2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
    }

    if hp_path:
        if hp_path.startswith("s3://"):
            bucket, key = hp_path.replace("s3://", "").split("/", 1)
            obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            overrides = json.loads(obj["Body"].read())
        else:
            with open(hp_path, "r") as f:
                overrides = json.load(f)
        params.update(overrides)

    params["num_class"] = len(pd.unique(y_train))

    evals = [(dval, "validation")] if dval is not None else [(dtrain, "train")]

    bst = xgb.train(
        params,
        dtrain,
        evals=evals,
        num_boost_round=300,
        early_stopping_rounds=20 if dval is not None else None,
    )

    os.makedirs(output_dir, exist_ok=True)
    bst.save_model(os.path.join(output_dir, "model.xgb"))

    fe = joblib.load(fe_path)
    joblib.dump(fe, os.path.join(output_dir, "fe.pkl"))

    le = joblib.load(label_encoder_path)
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    # Keep inference entrypoint in the artifact for convenience
    inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
    if os.path.exists(inference_src):
        shutil.copy(inference_src, os.path.join(output_dir, "inference.py"))

    feature_encoder_src = os.path.join(os.path.dirname(__file__), "feature_encoder.py")
    if os.path.exists(feature_encoder_src):
        shutil.copy(feature_encoder_src, os.path.join(output_dir, "feature_encoder.py"))


if __name__ == "__main__":
    main()
