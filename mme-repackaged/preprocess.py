import json
import boto3
import time

sm = boto3.client("sagemaker")

def handler(event, context):
    target = event["Target"]
    train_s3 = event["TrainS3"]
    val_s3   = event["ValS3"]
    role_arn = event["RoleArn"]
    output_path = event["OutputPath"]

    job_name = f"automl-v2-{target}-{int(time.time())}"

    # ---- Create AutoML V2 job
    response = sm.create_auto_ml_job_v2(
        AutoMLJobName=job_name,
        AutoMLJobInputDataConfig=[
            {
                "DataSource": {
                    "S3DataSource": {"S3Uri": train_s3}
                },
                "TargetAttributeName": target,
                "ChannelType": "training"
            },
            {
                "DataSource": {
                    "S3DataSource": {"S3Uri": val_s3}
                },
                "TargetAttributeName": target,
                "ChannelType": "validation"
            }
        ],
        OutputDataConfig={"S3OutputPath": output_path},
        AutoMLProblemTypeConfig={
            "TabularJobConfig": {
                "CompletionCriteria": {"MaxCandidates": 5},
                "CandidateGenerationConfig": {
                    "AlgorithmsConfig": [
                        {"XGBoost": {}}     # ⭐ FORCE XGBOOST ONLY
                    ]
                }
            }
        },
        RoleArn=role_arn,
    )

    # ---- Wait for the job to complete
    status = ""
    while status not in ("Completed", "Failed", "Stopped"):
        time.sleep(30)
        desc = sm.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        status = desc["AutoMLJobStatus"]
        print("Status:", status)

    if status != "Completed":
        raise Exception(f"AUTOML FAILED: {status}")

    best = desc["BestCandidate"]
    model_data = best["InferenceContainer"]["ModelDataUrl"]
    image_uri  = best["InferenceContainer"]["Image"]

    return {
        "ModelDataUrl": model_data,
        "ImageUri": image_uri
    }
