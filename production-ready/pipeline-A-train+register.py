# =========================
# Cell 1: Setup variables
# =========================

import os
import boto3
import sagemaker
from urllib.parse import urlparse
from botocore.exceptions import ClientError

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
role_arn = sagemaker.get_execution_role()

# ---- Client/project knobs ----
CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-mlops"
OUTPUT_PREFIX = "mlops"   # root prefix for all ML artifacts

# ---- Input data ----
BUCKET        = sm_sess.default_bucket()
INPUT_S3CSV   = f"s3://{BUCKET}/input/data.csv"

# ---- Targets (per target model) ----
TARGET_COLS = [
    "DepartmentCode",
    "AccountCode",
    "SubAccountCode",
    "LocationCode"
]

# ---- Input features to keep ----
INPUT_FEATURES = [
    "VendorName",
    "LineDescription",
    "ClubNumber"
]

# ---- Split parameters (per-target) ----
VAL_FRAC_DEFAULT        = 0.20
MIN_SUPPORT_DEFAULT     = 5
RARE_TRAIN_ONLY_DEFAULT = True

def _parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")

# Validate input CSV exists
s3 = boto3.client("s3", region_name=region)
csv_bucket, csv_key = _parse_s3(INPUT_S3CSV)
try:
    s3.head_object(Bucket=csv_bucket, Key=csv_key)
except ClientError as e:
    raise FileNotFoundError(f"CSV not found at {INPUT_S3CSV}") from e

print("Region:", region)
print("Role:", role_arn)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Targets:", TARGET_COLS)
print("Input features:", INPUT_FEATURES)
print("Project:", PROJECT_NAME, "| Output prefix:", OUTPUT_PREFIX)

# ============================================================
# Cell 2: Write processing script — independent per-target splits
# ============================================================

import os, textwrap

split_script = textwrap.dedent("""
import argparse, os, glob
import pandas as pd
import numpy as np

def find_input_csv(mounted_dir: str) -> str:
    candidates = glob.glob(os.path.join(mounted_dir, "*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(f"No CSV found under {mounted_dir}")

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

    # Guarantee at least two classes in training set (if possible)
    if train[target_col].nunique(dropna=True) < 2 and counts.nunique() > 1:
        val_groups = val.groupby(target_col)
        for cls, g in val_groups:
            if cls not in set(train[target_col].unique()):
                idx = g.index[0]
                train = pd.concat([train, val.loc[[idx]]], axis=0)
                val = val.drop(index=[idx])
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

    local_csv = find_input_csv(args.mounted_input_dir)
    df = pd.read_csv(local_csv, low_memory=False)

    targets      = [c.strip() for c in args.targets_csv.split(",") if c.strip()]
    input_feats  = [c.strip() for c in args.input_features_csv.split(",") if c.strip()]

    # Validate columns
    for c in input_feats + targets:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV header")

    for tgt in targets:
        train_df, val_df, counts = per_target_split(
            df=df,
            target_col=tgt,
            input_feats=input_feats,
            val_frac=args.val_frac,
            seed=args.random_seed,
            min_support=args.min_support,
            rare_train_only=rare_train_only,
        )

        out_tr = os.path.join(args.output_dir, tgt, "train")
        out_va = os.path.join(args.output_dir, tgt, "validation")
        os.makedirs(out_tr, exist_ok=True)
        os.makedirs(out_va, exist_ok=True)

        train_df.to_csv(os.path.join(out_tr, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_va, "validation.csv"), index=False)

if __name__ == "__main__":
    main()
""").strip()

with open("prepare_per_target_splits.py", "w") as f:
    f.write(split_script)

print("Wrote prepare_per_target_splits.py")

# ============================================================
# Cell 3: Write SKLearn training/inference script (updated to copy inference.py into /code)
# ============================================================
import os, textwrap

os.makedirs("sklearn_src", exist_ok=True)

model_script = textwrap.dedent("""
import os
import argparse
import io
import json
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher
from scipy import sparse


HASH_SIZE_TEXT = 128
HASH_SIZE_CAT  = 32


def hash_text_series(series: pd.Series, n_features: int):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    data = series.fillna("__MISSING__").astype(str).tolist()
    sparse_matrix = hasher.transform([[x] for x in data])
    return sparse_matrix


def process_features(
    df: pd.DataFrame,
    is_training: bool = True,
    training_meta: Optional[dict] = None
):
    # Identify feature types
    if is_training:
        num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        other_cols = [c for c in df.columns if c not in num_cols]

        text_cols = [c for c in other_cols if "desc" in c.lower()]
        cat_cols  = [c for c in other_cols if c not in text_cols]

        meta = {
            "num_cols": num_cols,
            "text_cols": text_cols,
            "cat_cols": cat_cols,
        }
    else:
        meta      = training_meta
        num_cols  = meta["num_cols"]
        text_cols = meta["text_cols"]
        cat_cols  = meta["cat_cols"]

    # Numeric → sparse
    X_num = sparse.csr_matrix(df[num_cols].fillna(0).to_numpy(dtype=np.float32))

    # Categorical hashed
    X_cat_list = []
    for c in cat_cols:
        Xc = hash_text_series(df[c], HASH_SIZE_CAT)
        X_cat_list.append(Xc)

    # Text hashed
    X_text_list = []
    for c in text_cols:
        Xt = hash_text_series(df[c], HASH_SIZE_TEXT)
        X_text_list.append(Xt)

    parts = [X_num] + X_cat_list + X_text_list
    X = sparse.hstack(parts).tocsr()

    return X, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", type=str, required=True)
    args, _ = parser.parse_known_args()

    target_name = args.target_name

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_df = pd.read_csv(os.path.join(train_dir, "train.csv"), low_memory=False)
    val_df   = pd.read_csv(os.path.join(val_dir,  "validation.csv"), low_memory=False)

    y_train   = train_df[target_name].astype(str)
    X_train_df = train_df.drop(columns=[target_name])

    y_val    = val_df[target_name].astype(str)
    X_val_df = val_df.drop(columns=[target_name])

    X_train, fe_meta = process_features(df=X_train_df, is_training=True)
    X_val, _         = process_features(df=X_val_df, is_training=False, training_meta=fe_meta)

    print(f"[train] X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=20,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    if len(y_val) > 0:
        preds = clf.predict(X_val)
        acc = (preds == y_val).mean()
        print(f"[train] Validation accuracy ({target_name}): {acc:.4f}")

    bundle = {
        "model": clf,
        "fe_meta": fe_meta,
        "target_name": target_name,
    }

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(model_dir, "model.joblib"))
    print("[train] Saved model bundle to", model_dir)

    # ==============================
    # NEW: ensure inference.py is in model_dir/code for MME
    # ==============================
    import shutil

    # In SKLearn estimator, source_dir is placed under /opt/ml/code
    src_inference = os.path.join("/opt/ml/code", "inference.py")

    # For Multi-Model Endpoint, each model.tar.gz must contain: code/inference.py
    code_dir = os.path.join(model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    dst_inference = os.path.join(code_dir, "inference.py")

    if os.path.exists(src_inference):
        shutil.copy(src_inference, dst_inference)
        print(f"[train] Copied inference.py into {code_dir}")
    else:
        print(f"[train] WARNING: inference.py not found at {src_inference}")


# ============== Inference functions (MME-compatible) ==============

def model_fn(model_dir: str):
    path = os.path.join(model_dir, "model.joblib")
    bundle = joblib.load(path)
    return bundle


def input_fn(input_data, content_type: str):
    if content_type == "text/csv":
        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")
        return pd.read_csv(io.StringIO(input_data))
    if content_type == "application/json":
        obj = json.loads(input_data)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        return pd.DataFrame([obj])
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(data: pd.DataFrame, model_bundle):
    target   = model_bundle["target_name"]
    fe_meta  = model_bundle["fe_meta"]
    clf      = model_bundle["model"]

    if target in data.columns:
        data = data.drop(columns=[target])

    X, _ = process_features(df=data, is_training=False, training_meta=fe_meta)
    preds = clf.predict(X)
    return preds


def output_fn(prediction, accept: str):
    if hasattr(prediction, "tolist"):
        preds = prediction.tolist()
    else:
        preds = list(prediction)

    if accept == "text/csv":
        return ",".join(str(x) for x in preds), "text/csv"

    return json.dumps(preds), "application/json"
""")

with open("sklearn_src/model_script.py", "w") as f:
    f.write(model_script)

print("Wrote updated sklearn_src/model_script.py")

# ============================================================
# Cell X: Write inference.py into sklearn_src for MME serving
# ============================================================

import os, textwrap

inference_script = textwrap.dedent("""
import json
import joblib
import pandas as pd
import numpy as np
import io
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher


HASH_SIZE_TEXT = 128
HASH_SIZE_CAT  = 32


def hash_text_series(series, n_features):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    data = series.fillna("__MISSING__").astype(str).tolist()
    return hasher.transform([[x] for x in data])


def process_features(df, fe_meta):
    num_cols  = fe_meta["num_cols"]
    text_cols = fe_meta["text_cols"]
    cat_cols  = fe_meta["cat_cols"]

    X_num = sparse.csr_matrix(df[num_cols].fillna(0).to_numpy(dtype=np.float32))

    X_cat = [hash_text_series(df[c], HASH_SIZE_CAT) for c in cat_cols]
    X_txt = [hash_text_series(df[c], HASH_SIZE_TEXT) for c in text_cols]

    X = sparse.hstack([X_num] + X_cat + X_txt).tocsr()
    return X


def model_fn(model_dir):
    bundle = joblib.load(f"{model_dir}/model.joblib")
    return bundle


def input_fn(request_body, content_type):
    if content_type == "application/json":
        obj = json.loads(request_body)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        return pd.DataFrame([obj])

    if content_type == "text/csv":
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        return pd.read_csv(io.StringIO(request_body))

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(df, model_bundle):
    target = model_bundle["target_name"]
    fe_meta = model_bundle["fe_meta"]
    clf = model_bundle["model"]

    if target in df.columns:
        df = df.drop(columns=[target])

    X = process_features(df, fe_meta)
    preds = clf.predict(X)
    return preds


def output_fn(prediction, accept):
    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()
    return json.dumps({"prediction": prediction}), accept
""")

os.makedirs("sklearn_src", exist_ok=True)
with open("sklearn_src/inference.py", "w") as f:
    f.write(inference_script)

print("Wrote sklearn_src/inference.py")

# ============================================================
# Cell 4: Write evaluation script (per-target accuracy)
# ============================================================
import os, textwrap

eval_script = textwrap.dedent("""
import os
import json
import tarfile

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction import FeatureHasher
from scipy import sparse
from sklearn.metrics import accuracy_score

# ---- must match training script ----
HASH_SIZE_TEXT = 128
HASH_SIZE_CAT  = 32


def hash_text_series(series: pd.Series, n_features: int):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    data = series.fillna("__MISSING__").astype(str).tolist()
    sparse_matrix = hasher.transform([[x] for x in data])
    return sparse_matrix


def process_features(df: pd.DataFrame, is_training=True, training_meta=None):
    '''Same logic as in training, but without Python 3.10 union types.'''
    if is_training:
        num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        other_cols = [c for c in df.columns if c not in num_cols]

        text_cols = [c for c in other_cols if "desc" in c.lower()]
        cat_cols = [c for c in other_cols if c not in text_cols]

        meta = {
            "num_cols": num_cols,
            "text_cols": text_cols,
            "cat_cols": cat_cols,
        }
    else:
        meta = training_meta
        num_cols = meta["num_cols"]
        text_cols = meta["text_cols"]
        cat_cols = meta["cat_cols"]

    X_num = sparse.csr_matrix(df[num_cols].fillna(0).to_numpy(dtype=np.float32))

    X_cat_list = []
    for c in cat_cols:
        Xc = hash_text_series(df[c], HASH_SIZE_CAT)
        X_cat_list.append(Xc)

    X_text_list = []
    for c in text_cols:
        Xt = hash_text_series(df[c], HASH_SIZE_TEXT)
        X_text_list.append(Xt)

    parts = [X_num] + X_cat_list + X_text_list
    X = sparse.hstack(parts).tocsr()

    return X, meta


def load_bundle_from_tar(model_tar_path: str):
    work_dir = "/opt/ml/processing/model_artifacts"
    os.makedirs(work_dir, exist_ok=True)

    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=work_dir)

    bundle_path = os.path.join(work_dir, "model.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"model.joblib not found inside {model_tar_path}")

    bundle = joblib.load(bundle_path)
    return bundle


def main():
    model_dir   = "/opt/ml/processing/model"
    val_dir     = "/opt/ml/processing/validation"
    output_dir  = "/opt/ml/processing/output"

    # find model.tar.gz under model_dir
    candidates = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".tar.gz")]
    if not candidates:
        raise FileNotFoundError(f"No .tar.gz model found under {model_dir}")
    model_tar = candidates[0]

    bundle = load_bundle_from_tar(model_tar)
    clf      = bundle["model"]
    fe_meta  = bundle["fe_meta"]
    tgt      = bundle["target_name"]

    val_csv = os.path.join(val_dir, "validation.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"validation.csv not found at {val_csv}")

    val_df = pd.read_csv(val_csv, low_memory=False)

    if tgt not in val_df.columns:
        raise ValueError(f"Target column '{tgt}' not found in validation.csv")

    y_true    = val_df[tgt].astype(str)
    X_val_df  = val_df.drop(columns=[tgt])

    X_val, _ = process_features(df=X_val_df, is_training=False, training_meta=fe_meta)
    preds    = clf.predict(X_val)

    acc = float(accuracy_score(y_true, preds))

    # ---- SageMaker-compliant metrics format for Studio UI ----
    metrics = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": acc
            }
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"[eval] target={tgt} accuracy={acc:.4f} samples={len(y_true)}")


if __name__ == "__main__":
    main()
""").strip()

os.makedirs("sklearn_src", exist_ok=True)
with open("sklearn_src/evaluate_model.py", "w") as f:
    f.write(eval_script)

print("Wrote sklearn_src/evaluate_model.py")

# ============================================================
# Cell 5: Pipeline A — Train + Evaluate + Register (per target)
# ============================================================
import boto3
import sagemaker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.step_collections import RegisterModel

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
p_sess   = PipelineSession()
role_arn = sagemaker.get_execution_role()

# Reuse from Cell 1:
# CLIENT_NAME, PROJECT_NAME, OUTPUT_PREFIX, BUCKET, INPUT_S3CSV
# TARGET_COLS, INPUT_FEATURES

# -------- Pipeline parameters --------
bucket_param          = ParameterString("Bucket", default_value=BUCKET)
input_s3_csv_param    = ParameterString("InputS3CSV", default_value=INPUT_S3CSV)
val_frac_param        = ParameterFloat("ValFrac", default_value=0.20)
seed_param            = ParameterInteger("RandomSeed", default_value=42)
min_support_param     = ParameterInteger("MinSupport", default_value=5)
rare_train_only_param = ParameterBoolean("RareTrainOnly", default_value=True)

sklearn_version = "1.2-1"
sklearn_image   = sagemaker.image_uris.retrieve("sklearn", region, version=sklearn_version)
print("Using SKLearn image:", sklearn_image)

# -------- Step 1 — per-target splits --------
split_processor = ScriptProcessor(
    image_uri=sklearn_image,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.extend([
        ProcessingOutput(
            output_name=f"train_{tgt}",
            source=f"/opt/ml/processing/output/{tgt}/train",
        ),
        ProcessingOutput(
            output_name=f"validation_{tgt}",
            source=f"/opt/ml/processing/output/{tgt}/validation",
        ),
    ])

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=split_processor,
    inputs=[ProcessingInput(source=input_s3_csv_param, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac_param.to_string(),
        "--random_seed", seed_param.to_string(),
        "--min_support", min_support_param.to_string(),
        "--rare_train_only", rare_train_only_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

# -------- Step 2 & 3 — Train + Evaluate + Register per target --------
train_steps    = {}
eval_steps     = {}
register_steps = []

for tgt in TARGET_COLS:

    # ---- Training ----
    sklearn_estimator = SKLearn(
        entry_point="model_script.py",
        source_dir="sklearn_src",
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version=sklearn_version,
        sagemaker_session=p_sess,
        hyperparameters={"target-name": tgt},
    )

    train_step = TrainingStep(
        name=f"Train_{tgt}",
        estimator=sklearn_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=split_step.properties.ProcessingOutputConfig.Outputs[f"train_{tgt}"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{tgt}"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=CacheConfig(enable_caching=False),
    )
    train_steps[tgt] = train_step

    # ---- Evaluation ----
    eval_processor = ScriptProcessor(
        image_uri=sklearn_image,
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        command=["python3"],
        sagemaker_session=p_sess,
    )

    eval_step = ProcessingStep(
        name=f"Evaluate_{tgt}",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{tgt}"].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics",
                source="/opt/ml/processing/output",
            )
        ],
        code="sklearn_src/evaluate_model.py",
        cache_config=CacheConfig(enable_caching=False),
    )
    eval_steps[tgt] = eval_step

    # ---- Model Metrics for registry ----
    metrics_s3_uri = eval_step.properties.ProcessingOutputConfig.Outputs["metrics"].S3Output.S3Uri
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=metrics_s3_uri,
            content_type="application/json",
        )
    )

    # ---- Register in Model Registry ----
    model_package_group_name = f"{CLIENT_NAME}-{tgt}-models"

    register_step = RegisterModel(
        name=f"Register_{tgt}",
        estimator=sklearn_estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status="Approved",   # or "PendingManualApproval"
        image_uri=sklearn_image,
    )

    register_steps.append(register_step)

# -------- Assemble Pipeline A --------
pipeline_a = Pipeline(
    name=f"{PROJECT_NAME}-train-eval-register",
    parameters=[
        bucket_param,
        input_s3_csv_param,
        val_frac_param,
        seed_param,
        min_support_param,
        rare_train_only_param,
    ],
    steps=[split_step] + list(train_steps.values()) + list(eval_steps.values()) + register_steps,
    sagemaker_session=p_sess,
)

pipeline_a.upsert(role_arn=role_arn)
execution = pipeline_a.start()
print("Pipeline A execution started:", execution.arn)
