# ============================================================
# Cell 3: Helper scripts for AutoML search + config extraction + MME repack
# ============================================================
import textwrap

# ------------------------------------------------------------
# Script 1: run_automl_xgb.py  (AutoML hyperparameter search only)
# ------------------------------------------------------------
run_automl = textwrap.dedent("""
import boto3, json, time, argparse

def wait(sm, job_name):
    print(f"[INFO] Waiting for AutoML job: {job_name}")
    while True:
        desc = sm.describe_auto_ml_job(AutoMLJobName=job_name)
        status = desc["AutoMLJobStatus"]
        sec    = desc["AutoMLJobSecondaryStatus"]
        print(f"  Status: {status} / {sec}")
        if status in ("Failed", "Stopped"):
            raise RuntimeError(f"AutoML job {job_name} failed: {desc}")
        if status == "Completed":
            print("[INFO] AutoML Completed.")
            return desc
        time.sleep(20)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job_name", required=True)
    p.add_argument("--train_s3_uri", required=True)
    p.add_argument("--target_col", required=True)
    p.add_argument("--output_prefix", required=True)
    p.add_argument("--role", required=True)
    args = p.parse_args()

    sm = boto3.client("sagemaker")

    # Force AutoML to XGBoost only
    cfg = {
        "AlgorithmsConfig": [
            {"AutoMLAlgorithms": ["XGBOOST"]}
        ],
        "CompletionCriteria": {
            "MaxCandidates": 5
        }
    }

    print(f"[INFO] Starting AutoML XGBoost-only job {args.job_name}")

    sm.create_auto_ml_job(
        AutoMLJobName=args.job_name,
        AutoMLJobConfig=cfg,
        InputDataConfig=[
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": args.train_s3_uri,
                    }
                },
                "TargetAttributeName": args.target_col,
            }
        ],
        OutputDataConfig={"S3OutputPath": args.output_prefix},
        RoleArn=args.role,
        ProblemType="MulticlassClassification"
    )

    desc = wait(sm, args.job_name)

    best = desc["BestCandidate"]

    with open("automl_best.json", "w") as f:
        json.dump(best, f)

    # This is what the ProcessingStep will capture as output
    print("automl_best.json")

if __name__ == "__main__":
    main()
""").strip()

with open("run_automl_xgb.py", "w") as f:
    f.write(run_automl)


# ------------------------------------------------------------
# Script 2: extract_automl_config.py
# Extract hyperparameters + feature engineering config
# ------------------------------------------------------------
extract_automl_config = textwrap.dedent("""
import json, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_json", required=True)
    args = p.parse_args()

    with open(args.best_json, "r") as f:
        best = json.load(f)

    # Hyperparameters chosen by AutoML
    hp = best["CandidateProperties"].get("CandidateMetrics", [{}])[0].get("HyperParameters", {})

    # Feature-engineering transformation map (if present)
    fe = best["CandidateProperties"].get("FeatureEngineering", {})

    with open("hyperparams.json", "w") as f:
        json.dump(hp, f)

    with open("feature_eng.json", "w") as f:
        json.dump(fe, f)

    print("hyperparams.json")
    print("feature_eng.json")

if __name__ == "__main__":
    main()
""").strip()

with open("extract_automl_config.py", "w") as f:
    f.write(extract_automl_config)


# ------------------------------------------------------------
# Script 3: inference.py  (MME-compatible XGBoost handler)
# ------------------------------------------------------------
inference_script = textwrap.dedent("""
import json, os
import xgboost as xgb
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.xgb")
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

def input_fn(request_body, request_content_type):
    # Request is JSON: {"col1": [...], "col2": [...]} or a list of dicts
    payload = json.loads(request_body)
    df = pd.DataFrame(payload)
    return df

def predict_fn(input_data, model):
    dtest = xgb.DMatrix(input_data)
    preds = model.predict(dtest)
    return preds.tolist()

def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
""").strip()

with open("inference.py", "w") as f:
    f.write(inference_script)


# ------------------------------------------------------------
# Script 4: repack_for_mme.py
# Repack XGB model + inference.py → model_mme.tar.gz
# ------------------------------------------------------------
repack_script = textwrap.dedent("""
import argparse, tarfile, shutil, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trained_model_s3", required=True)
    p.add_argument("--inference_script", default="inference.py")
    args = p.parse_args()

    # Make local workspace
    os.makedirs("model", exist_ok=True)

    # Auto-download SageMaker model.tar.gz
    import boto3
    s3 = boto3.client("s3")

    bucket, key = args.trained_model_s3.replace("s3://","").split("/",1)
    local_tar = "model.tar.gz"
    s3.download_file(bucket, key, local_tar)

    # Extract
    with tarfile.open(local_tar, "r:gz") as t:
        t.extractall("model")

    # Copy inference script
    shutil.copy(args.inference_script, "model/inference.py")

    # Write final tar
    out_path = "model_mme.tar.gz"
    with tarfile.open(out_path, "w:gz") as t:
        t.add("model", arcname=".")
    print(out_path)

if __name__ == "__main__":
    main()
""").strip()

with open("repack_for_mme.py", "w") as f:
    f.write(repack_script)

print("Scripts written:")
print("- run_automl_xgb.py")
print("- extract_automl_config.py")
print("- inference.py")
print("- repack_for_mme.py")
