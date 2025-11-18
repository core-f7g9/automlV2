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

# ============================================================
# Cell 2b: Write repack script for MME inference
# ============================================================
repack_script = textwrap.dedent("""
import argparse
import glob
import os
import shutil
import tarfile
import tempfile
import textwrap
import json
import subprocess
import sys

INFERENCE_TEMPLATE = \"\"\"
import os, site
import csv
import io
import json

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DEPS_DIR = os.path.join(BASE_DIR, "dependencies")
if os.path.isdir(DEPS_DIR):
    site.addsitedir(DEPS_DIR)

FEATURE_COLUMNS = {feature_list}
TARGET_NAME = "{target_name}"

def _find_model_file(model_dir):
    for root, _, files in os.walk(model_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith((".pkl", ".pickle", ".joblib")):
                return os.path.join(root, name)
    raise FileNotFoundError("Could not locate serialized model file under {{}}".format(model_dir))

def model_fn(model_dir):
    model_path = _find_model_file(model_dir)
    return joblib.load(model_path)

def input_fn(request_body, content_type):
    if content_type != "text/csv":
        raise ValueError("Unsupported content type: {{}}".format(content_type))
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    reader = csv.reader(io.StringIO(request_body))
    row = next(reader)
    expected = len(FEATURE_COLUMNS)
    if len(row) != expected:
        raise ValueError("Expected {{}} features, got {{}}".format(expected, len(row)))
    parsed = []
    for val in row:
        if val in ("", None):
            parsed.append(np.nan)
            continue
        try:
            parsed.append(float(val))
            continue
        except ValueError:
            parsed.append(val)
    frame = pd.DataFrame([parsed], columns=FEATURE_COLUMNS)
    return frame

def predict_fn(input_data, model):
    result = {{"target": TARGET_NAME}}
    preds = model.predict(input_data)
    result["label"] = preds[0]
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_data)[0]
            classes = getattr(model, "classes_", list(range(len(probs))))
            result["probabilities"] = {{
                str(cls): float(prob) for cls, prob in zip(classes, probs)
            }}
        except Exception:
            pass
    return result

def output_fn(prediction, accept):
    accept = accept or "application/json"
    return json.dumps(prediction), accept
\"\"\"

def build_inference_script(target_name, features):
    return textwrap.dedent(INFERENCE_TEMPLATE).format(
        target_name=target_name,
        feature_list=json.dumps(features)
    )

def find_model_tar(input_dir):
    candidates = sorted(glob.glob(os.path.join(input_dir, "*.tar.gz")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar.gz"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No .tar.gz artifacts found under {input_dir}")
    return candidates[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input/model")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output/repacked_model")
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--feature_list_csv", type=str, required=True)
    parser.add_argument("--inference_filename", type=str, default="inference.py")
    args = parser.parse_args()

    features = [c.strip() for c in args.feature_list_csv.split(",") if c.strip()]
    if not features:
        raise ValueError("At least one feature is required for inference")

    script_text = build_inference_script(args.target_name, features)
    src_tar = find_model_tar(args.input_dir)

    work_dir = tempfile.mkdtemp(prefix="repack-")
    try:
        with tarfile.open(src_tar, "r:gz") as tar:
            tar.extractall(work_dir)

        code_dir = os.path.join(work_dir, "code")
        os.makedirs(code_dir, exist_ok=True)

        # Put inference and requirements under code/ as expected by the SKLearn serving container
        script_path = os.path.join(code_dir, args.inference_filename)
        with open(script_path, "w") as f:
            f.write(script_text)

        # Keep dependencies explicit for the inference container
        req_path = os.path.join(code_dir, "requirements.txt")
        reqs = "\\n".join([
            "pandas",
            "numpy",
            "joblib",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
            "boto3",
            "botocore",
        ]) + "\\n"
        with open(req_path, "w") as f:
            f.write(reqs)

        # Vendor dependencies into the artifact so the container doesn't need internet
        deps_dir = os.path.join(code_dir, "dependencies")
        os.makedirs(deps_dir, exist_ok=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_path, "-t", deps_dir])

        # Explicitly tell the MMS runtime which module to import for this model
        with open(os.path.join(work_dir, ".sagemaker-inference.json"), "w") as f:
            json.dump({"program": args.inference_filename}, f)

        os.makedirs(args.output_dir, exist_ok=True)
        dst_tar = os.path.join(args.output_dir, "model.tar.gz")
        with tarfile.open(dst_tar, "w:gz") as tar:
            for root, _, files in os.walk(work_dir):
                for name in sorted(files):
                    path = os.path.join(root, name)
                    arcname = os.path.relpath(path, start=work_dir)
                    tar.add(path, arcname=arcname)

        print(f"Repacked model for target {args.target_name} -> {dst_tar}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
""").strip()

with open("repack_for_mme.py", "w") as f:
    f.write(repack_script)

print("Wrote repack_for_mme.py")
