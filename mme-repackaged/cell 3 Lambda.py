# ============================
# Cell 3: Write AutoML V2 Lambda (XGBoost-only)
# ============================

import textwrap

automl_lambda_code = textwrap.dedent("""
import json
import boto3
import time

sm = boto3.client("sagemaker")

def handler(event, context):
    target      = event["Target"]
    train_s3    = event["TrainS3"]
    val_s3      = event["ValS3"]
    role_arn    = event["RoleArn"]
    output_path = event["OutputPath"]

    job_name = f"automl-v2-{target}-{int(time.time())}"

    # Create AutoML V2 job
    sm.create_auto_ml_job_v2(
        AutoMLJobName=job_name,
        AutoMLJobInputDataConfig=[
            {
                "ChannelType": "training",
                "TargetAttributeName": target,
                "DataSource": {"S3DataSource": {"S3Uri": train_s3}}
            },
            {
                "ChannelType": "validation",
                "TargetAttributeName": target,
                "DataSource": {"S3DataSource": {"S3Uri": val_s3}}
            }
        ],
        OutputDataConfig={"S3OutputPath": output_path},
        AutoMLProblemTypeConfig={
            "TabularJobConfig": {
                "CompletionCriteria": {"MaxCandidates": 5},
                "CandidateGenerationConfig": {
                    "AlgorithmsConfig": [
                        {"XGBoost": {}}
                    ]
                }
            }
        },
        RoleArn=role_arn,
    )

    # Wait for completion
    status = ""
    while status not in ("Completed", "Failed", "Stopped"):
        time.sleep(30)
        desc = sm.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        status = desc["AutoMLJobStatus"]
        print("Status:", status)

    if status != "Completed":
        raise Exception(f"AUTOML FAILED: {status}")

    best = desc["BestCandidate"]

    # ⭐ Correct AutoML V2 structure:
    container = best["InferenceContainers"][0]
    model_data = container["ModelDataUrl"]
    image_uri  = container["ImageUri"]

    return {
        "ModelDataUrl": model_data,
        "ImageUri": image_uri,
        "AutoMLJobName": job_name
    }
""")

with open("lambda_automl_v2.py", "w") as f:
    f.write(automl_lambda_code)

print("Wrote lambda_automl_v2.py")
