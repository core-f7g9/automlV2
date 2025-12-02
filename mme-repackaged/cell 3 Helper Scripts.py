# ============================================================
# Cell 3 — Create all helper scripts: Lambda + repack + inference
# ============================================================

import os
import textwrap

# ============================================================
# 1) Lambda: launch AutoML V1 job and return best model artifact
# ============================================================
automl_lambda_code = textwrap.dedent("""
import json, time, boto3, os

sm = boto3.client("sagemaker")

def handler(event, context):
    config = event["automl_config"]
    name   = config["AutoMLJobName"]

    # Submit
    sm.create_auto_ml_job(**config)

    # Poll until finished
    while True:
        desc = sm.describe_auto_ml_job(AutoMLJobName=name)
        status = desc["AutoMLJobStatus"]
        if status in ("Completed","Failed","Stopped"):
            break
        time.sleep(20)

    if status != "Completed":
        raise Exception(f"AutoML failed with status {status}")

    best = desc["BestCandidate"]
    model_artifact = best["InferenceContainers"][0]["ModelDataUrl"]

    return {
        "best_model_artifact": model_artifact,
        "candidate_name": best["CandidateName"]
    }
""").strip()

with open("lambda_launch_automl.py", "w") as f:
    f.write(automl_lambda_code)

print("Wrote lambda_launch_automl.py")

# ============================================================
# 2) Lambda: perform repack for MME (copies artifact + adds inference.py)
# ============================================================
repack_lambda_code = textwrap.dedent("""
import json, tarfile, boto3, os, tempfile, shutil

s3 = boto3.client("s3")

def handler(event, context):
    model_s3 = event["model_artifact"]
    target   = event["target"]

    # Parse S3 URI
    assert model_s3.startswith("s3://")
    _, _, bucket_key = model_s3.partition("s3://")
    bucket, _, key = bucket_key.partition("/")

    # Download original artifact
    tmp = tempfile.mkdtemp()
    local_tar = os.path.join(tmp, "model.tar.gz")
    s3.download_file(bucket, key, local_tar)

    # Extract
    extract_dir = os.path.join(tmp, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as t:
        t.extractall(extract_dir)

    # Copy inference.py into artifact root
    shutil.copy("/opt/ml/processing/input/code/inference.py",
                os.path.join(extract_dir, "inference.py"))

    # Repack new tar
    repacked_tar = os.path.join(tmp, "repacked.tar.gz")
    with tarfile.open(repacked_tar, "w:gz") as t:
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                fp = os.path.join(root, f)
                arc = os.path.relpath(fp, extract_dir)
                t.add(fp, arcname=arc)

    # Upload new tar
    out_key = f"mlops/repacked/{target}/model.tar.gz"
    s3.upload_file(repacked_tar, bucket, out_key)

    out_uri = f"s3://{bucket}/{out_key}"
    return {"repacked_artifact": out_uri}
""").strip()

with open("lambda_repack_mme.py", "w") as f:
    f.write(repack_lambda_code)

print("Wrote lambda_repack_mme.py")

# ============================================================
# 3) inference.py — generic script required by MME & AutoGluon
# ============================================================
inference_code = textwrap.dedent("""
import os
import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb

# Generic inference handler so MME loads the correct model
# and supports text input via CSV-like payload.

def model_fn(model_dir):
    # XGBoost model expected at model.xgboost or similar
    # AutoML V1 XGB artifacts use 'model.pkl'
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        alt = os.path.join(model_dir, "model.xgboost")
        if os.path.exists(alt):
            model_path = alt
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(raw, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(pd.compat.StringIO(raw))
        return df
    raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(data, model):
    # AutoGluon XGB wrapper supports model.predict_proba
    preds = model.predict_proba(data)
    out = preds.tolist()
    return out

def output_fn(prediction, accept):
    return json.dumps(prediction)
""").strip()

with open("inference.py", "w") as f:
    f.write(inference_code)

print("Wrote inference.py")

# ============================================================
# 4) Store all scripts inside a folder for lambda packaging
# ============================================================
os.makedirs("lambda_code", exist_ok=True)
shutil_cmd = "cp lambda_launch_automl.py lambda_code/; cp lambda_repack_mme.py lambda_code/"
os.system(shutil_cmd)

print("Finished Cell 3 script generation.")
