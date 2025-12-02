# ============================================================
# Cell 3: Write all helper scripts for AutoML + best-candidate extraction + MME repack
# ============================================================
import textwrap

# ------------------------------------------------------------
# Script 1: run_automl_xgb.py 
# ------------------------------------------------------------
run_automl = textwrap.dedent("""
import boto3, json, time, argparse, os

def wait(sm, job_name):
    print(f"[INFO] Waiting for AutoML job: {job_name}")
    while True:
        resp = sm.describe_auto_ml_job(AutoMLJobName=job_name)
        status = resp["AutoMLJobStatus"]
        sec = resp["AutoMLJobSecondaryStatus"]
        print(f"  Status: {status} / {sec}")
        if status in ("Failed","Stopped"):
            raise RuntimeError(f"AutoML job {job_name} failed.")
        if status == "Completed":
            print("[INFO] AutoML Completed.")
            return resp
        time.sleep(30)

def get_best_candidate(desc):
    cands = desc.get("BestCandidate", None)
    if not cands:
        raise RuntimeError("No BestCandidate found in AutoML response.")
    return cands

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job_name", type=str, required=True)
    p.add_argument("--train_s3_uri", type=str, required=True)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--role_arn", type=str, required=True)
    p.add_argument("--output_prefix", type=str, required=True)
    args = p.parse_args()

    sm = boto3.client("sagemaker")

    # Force AutoML to XGBoost only
    cfg = {
        "CompletionCriteria": {
            "MaxCandidates": 5,
            "MaxAutoMLJobRuntimeInSeconds": 3600
        },
        "SecurityConfig": {},
        "AlgorithmsConfig": [
            {"AutoMLAlgorithms": ["XGBOOST"]}
        ]
    }

    print("[INFO] Starting AutoML XGBoost-only job:", args.job_name)

    sm.create_auto_ml_job(
        AutoMLJobName=args.job_name,
        AutoMLJobConfig=cfg,
        InputDataConfig=[
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": args.train_s3_uri
                    }
                },
                "TargetAttributeName": args.target_col
            }
        ],
        OutputDataConfig={"S3OutputPath": args.output_prefix},
        RoleArn=args.role_arn,
        ProblemType="MulticlassClassification"
    )

    desc = wait(sm, args.job_name)
    best = get_best_candidate(desc)

    # Save best candidate JSON
    out_path = "best_candidate.json"
    with open(out_path, "w") as f:
        json.dump(best, f)
    print("[INFO] Wrote best_candidate.json")

    # Pass path to pipeline
    print(out_path)

if __name__ == "__main__":
    main()
""").strip()

with open("run_automl_xgb.py", "w") as f:
    f.write(run_automl)


# ------------------------------------------------------------
# Script 2: extract_best_candidate.py
# ------------------------------------------------------------
extract_best = textwrap.dedent("""
import boto3, json, argparse, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_json", type=str, required=True)
    args = p.parse_args()

    with open(args.best_json, "r") as f:
        best = json.load(f)

    model_artifact = best["InferenceContainers"][0]["ModelDataUrl"]
    image_uri      = best["InferenceContainers"][0]["Image"]

    # Write to text files for downstream step
    with open("model_artifact.txt","w") as f:
        f.write(model_artifact)
    with open("image_uri.txt","w") as f:
        f.write(image_uri)

    print("model_artifact.txt")
    print("image_uri.txt")

if __name__ == "__main__":
    main()
""").strip()

with open("extract_best_candidate.py","w") as f:
    f.write(extract_best)


# ------------------------------------------------------------
# Script 3: inference.py (generic handler)
# ------------------------------------------------------------
inference_script = textwrap.dedent("""
import os, json, pickle
import numpy as np
import pandas as pd

def model_fn(model_dir):
    # Load XGBoost or sklearn model
    import joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction.tolist())
    raise ValueError("Unsupported response content type: {}".format(response_content_type))
""").strip()

with open("inference.py","w") as f:
    f.write(inference_script)


# ------------------------------------------------------------
# Script 4: repack_for_mme.py
# ------------------------------------------------------------
repack = textwrap.dedent("""
import tarfile, argparse, os, shutil

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_artifact", type=str, required=True)
    p.add_argument("--inference_script", type=str, default="inference.py")
    args = p.parse_args()

    # Extract AutoML model tar
    os.makedirs("model", exist_ok=True)
    with tarfile.open(args.model_artifact, "r:gz") as t:
        t.extractall("model")

    # Copy inference script
    shutil.copy(args.inference_script, "model/inference.py")

    # Repack for MME
    out_path = "model_mme.tar.gz"
    with tarfile.open(out_path, "w:gz") as t:
        t.add("model", arcname=".")
    print(out_path)

if __name__ == "__main__":
    main()
""").strip()

with open("repack_for_mme.py","w") as f:
    f.write(repack)

print("Scripts written: run_automl_xgb.py, extract_best_candidate.py, inference.py, repack_for_mme.py")
