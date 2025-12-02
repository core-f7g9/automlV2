import textwrap

train_code = textwrap.dedent("""
import argparse
import pandas as pd
import xgboost as xgb
import joblib
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--fe_model", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)

    y_train = df_train["target"]
    y_val   = df_val["target"]

    X_train = df_train.drop(columns=["target"])
    X_val   = df_val.drop(columns=["target"])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    params = {
        "objective": "multi:softprob",
        "num_class": len(y_train.unique()),
        "max_depth": 6,
        "eta": 0.2,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    bst = xgb.train(
        params,
        dtrain,
        evals=[(dval, "validation")],
        num_boost_round=300,
        early_stopping_rounds=20
    )

    # Save model + FE pipeline
    bst.save_model(os.path.join(args.output_dir, "model.xgb"))
    joblib.dump(joblib.load(args.fe_model), os.path.join(args.output_dir, "fe.pkl"))

if __name__ == "__main__":
    main()
""")

with open("train_xgb.py", "w") as f:
    f.write(train_code)

print("Wrote train_xgb.py")

import textwrap

infer_code = textwrap.dedent("""
import json
import os
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd

def model_fn(model_dir):
    fe = joblib.load(os.path.join(model_dir, "fe.pkl"))
    bst = xgb.Booster()
    bst.load_model(os.path.join(model_dir, "model.xgb"))
    return (fe, bst)

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    return pd.DataFrame(data)

def predict_fn(input_data, model):
    fe, bst = model
    X = fe.transform(input_data)
    dmat = xgb.DMatrix(X)
    preds = bst.predict(dmat)
    return preds.tolist()

def output_fn(prediction, content_type):
    return json.dumps(prediction)
""")

with open("inference.py", "w") as f:
    f.write(infer_code)

print("Wrote inference.py")

import textwrap

repack_code = textwrap.dedent("""
import argparse
import os
import tarfile
import shutil

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_tar", required=True)
    args = p.parse_args()

    with tarfile.open(args.output_tar, "w:gz") as tar:
        tar.add(args.model_dir, arcname=".")

if __name__ == "__main__":
    main()
""")

with open("repack_for_mme.py", "w") as f:
    f.write(repack_code)

print("Wrote repack_for_mme.py")
