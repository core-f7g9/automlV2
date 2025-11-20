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
import textwrap, os

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
from typing import Tuple

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = CatBoostRegressor = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None

BASE_DIR = os.path.dirname(__file__)
DEPS_DIR = os.path.join(BASE_DIR, "dependencies")
if os.path.isdir(DEPS_DIR):
    site.addsitedir(DEPS_DIR)

FEATURE_COLUMNS = {feature_list}
TARGET_NAME = "{target_name}"

MODEL_PRIORITY = (
    ("pipeline.pkl", "joblib"),
    ("model.pkl", "joblib"),
    ("model.joblib", "joblib"),
    ("model.pickle", "joblib"),
    ("model.bin", "joblib"),
    ("model.json", "xgboost"),
    ("xgboost-model", "xgboost"),
    ("model.cbm", "catboost"),
)

def _flatten_model_paths(model_dir):
    discovered = []
    for root, _, files in os.walk(model_dir):
        for name in files:
            discovered.append((os.path.join(root, name), name.lower()))
    return discovered

def _find_model_file(model_dir) -> Tuple[str, str]:
    discovered = _flatten_model_paths(model_dir)
    for candidate, kind in MODEL_PRIORITY:
        for path, lower in discovered:
            if lower == candidate or lower.endswith(candidate):
                return path, kind

    for path, lower in discovered:
        if lower.endswith((".pkl", ".pickle", ".joblib")):
            return path, "joblib"
        if lower == "xgboost-model":
            return path, "xgboost"
        if lower.endswith(".cbm"):
            return path, "catboost"
    raise FileNotFoundError("Could not locate serialized model file under {{}}".format(model_dir))

def _load_model(path, kind_hint):
    lower = os.path.basename(path).lower()
    if kind_hint == "xgboost" or lower in ("xgboost-model", "model.json", "model.bin"):
        if xgb is None:
            raise ImportError("xgboost is required to load the model but is not available in this container")
        booster = xgb.Booster()
        booster.load_model(path)
        return "xgboost_booster", booster

    if kind_hint == "catboost" or lower.endswith(".cbm"):
        if CatBoostClassifier is None:
            raise ImportError("catboost is required to load the model but is not available in this container")
        try:
            model = CatBoostClassifier()
            model.load_model(path)
            return "catboost_classifier", model
        except Exception:
            model = CatBoostRegressor()
            model.load_model(path)
            return "catboost_regressor", model

    obj = joblib.load(path)
    if TabularPredictor is not None and isinstance(obj, TabularPredictor):
        return "autogluon", obj
    return "joblib", obj

def model_fn(model_dir):
    model_path, kind = _find_model_file(model_dir)
    kind, model = _load_model(model_path, kind)
    return {{"kind": kind, "model": model}}

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
    kind = model.get("kind")
    model = model.get("model")

    if kind == "autogluon":
        preds = model.predict(input_data)
        label = preds.iloc[0] if hasattr(preds, "iloc") else preds[0]
        result = {{"target": TARGET_NAME, "label": label}}
        if hasattr(model, "predict_proba"):
            try:
                prob_df = model.predict_proba(input_data)
                probs = prob_df.iloc[0].to_dict() if hasattr(prob_df, "iloc") else prob_df[0]
                result["probabilities"] = {{str(k): float(v) for k, v in probs.items()}}
            except Exception:
                pass
        return result

    if kind == "xgboost_booster":
        if xgb is None:
            raise ImportError("xgboost is required for inference but is not available in this container")
        non_numeric = [c for c in input_data.columns if not np.issubdtype(input_data[c].dtype, np.number)]
        if non_numeric:
            raise ValueError(f"Loaded a bare XGBoost booster but input has non-numeric columns: {{non_numeric}}. Re-run training to ensure a pipeline.pkl exists.")
        dmat = xgb.DMatrix(input_data)
        raw = np.array(model.predict(dmat))
        if raw.ndim > 1 and raw.shape[1] > 1:
            label_idx = int(np.argmax(raw[0]))
            probs = {{str(i): float(v) for i, v in enumerate(raw[0].tolist())}}
            return {{"target": TARGET_NAME, "label": label_idx, "probabilities": probs}}
        return {{"target": TARGET_NAME, "label": float(raw[0])}}

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
    parser.add_argument("--vendor_dependencies", action="store_true", help="pip install common ML deps into the artifact")
    args = parser.parse_args()

    features = [c.strip() for c in args.feature_list_csv.split(",") if c.strip()]
    if not features:
        raise ValueError("At least one feature is required for inference")

    script_text = build_inference_script(args.target_name, features)
    src_tar = find_model_tar(args.input_dir)

    work_dir = tempfile.mkdtemp(prefix="repack-")
    try:
        # Extract original Autopilot model.tar.gz
        with tarfile.open(src_tar, "r:gz") as tar:
            tar.extractall(work_dir)

        # Place inference.py at the ROOT of the model directory
        script_path = os.path.join(work_dir, args.inference_filename)
        with open(script_path, "w") as f:
            f.write(script_text)

        if args.vendor_dependencies:
            # requirements.txt with only what we actually need
            req_path = os.path.join(work_dir, "requirements.txt")
            reqs = "\\n".join([
                "pandas",
                "numpy",
                "joblib",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "catboost",
            ]) + "\\n"
            with open(req_path, "w") as f:
                f.write(reqs)

            # Vendor dependencies into 'dependencies' so container doesn't need internet
            deps_dir = os.path.join(work_dir, "dependencies")
            os.makedirs(deps_dir, exist_ok=True)
            subprocess.check_call([
                sys.executable,
                "-m", "pip", "install",
                "--no-cache-dir",
                "-r", req_path,
                "-t", deps_dir
            ])

        # Let the runtime know our entrypoint; path is relative to this root
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
