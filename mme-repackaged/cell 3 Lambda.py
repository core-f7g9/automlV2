# ============================
# Cell 3 — Write training script
# ============================

import textwrap

train_script = textwrap.dedent("""
import argparse
import os
import json
import numpy as np
import joblib
import xgboost as xgb
from scipy import sparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()
    tgt = args.target

    # Input dirs (SageMaker Training)
    model_dir = os.environ.get("SM_MODEL_DIR")
    data_dir  = os.environ.get("SM_CHANNEL_DATA")

    # Load matrices
    Xtr = sparse.load_npz(os.path.join(data_dir, "X_train.npz"))
    Xva = sparse.load_npz(os.path.join(data_dir, "X_val.npz"))
    ytr = np.load(os.path.join(data_dir, "y_train.npy"))
    yva = np.load(os.path.join(data_dir, "y_val.npy"))

    # Load preprocessing artifacts
    tfidf_word = joblib.load(os.path.join(data_dir, "tfidf_word.pkl"))
    tfidf_char = joblib.load(os.path.join(data_dir, "tfidf_char.pkl"))
    le         = joblib.load(os.path.join(data_dir, "label_encoder.pkl"))
    config     = json.load(open(os.path.join(data_dir, "config.json")))

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xva, label=yva)

    params = {
        "objective": "multi:softprob",
        "num_class": len(le.classes_),
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "eta": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    booster = xgb.train(
        params,
        dtrain,
        evals=[(dvalid, "validation")],
        num_boost_round=300,
        early_stopping_rounds=20
    )

    # Save model + preprocess artifacts
    os.makedirs(model_dir, exist_ok=True)

    booster.save_model(os.path.join(model_dir, "xgb-model.json"))
    joblib.dump(tfidf_word, os.path.join(model_dir, "tfidf_word.pkl"))
    joblib.dump(tfidf_char, os.path.join(model_dir, "tfidf_char.pkl"))
    joblib.dump(le,         os.path.join(model_dir, "label_encoder.pkl"))

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Copy inference.py
    import shutil
    shutil.copy("/opt/ml/code/inference.py", os.path.join(model_dir, "inference.py"))

    print("Training complete for:", tgt)

if __name__ == "__main__":
    main()
""")

with open("train_xgb.py", "w") as f:
    f.write(train_script)

print("Wrote train_xgb.py")
