# deploy_xgb_mme.py
import os
import time
import json
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, WaiterError


def _parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def handler(event, context):
    """
    Lambda entrypoint used by the SageMaker Pipeline LambdaStep.

    Expected event keys (from LambdaStep inputs):
      - EndpointName
      - InstanceType
      - InitialInstanceCount
      - ModelsPrefix       (s3://.../mme/client1/models/)
      - TargetNamesCSV     (e.g. "DepartmentCode,AccountCode,...")
      - TargetModelDatasCSV (CSV of S3 URIs to model.tar.gz from training)
      - XGBoostImage
    """

    print("Received event:", json.dumps(event))

    endpoint_name = event["EndpointName"]
    instance_type = event["InstanceType"]
    initial_instance_count = int(event["InitialInstanceCount"])
    models_prefix_uri = event["ModelsPrefix"]
    target_names_csv = event["TargetNamesCSV"]
    target_models_csv = event["TargetModelDatasCSV"]
    image_uri = event["XGBoostImage"]

    # Execution role for the SageMaker model
    exec_role_arn = os.environ.get("EXEC_ROLE_ARN")
    if not exec_role_arn:
        raise RuntimeError("EXEC_ROLE_ARN environment variable not set")

    sm = boto3.client("sagemaker")
    s3 = boto3.client("s3")

    target_names = [t.strip() for t in target_names_csv.split(",") if t.strip()]
    model_uris = [m.strip() for m in target_models_csv.split(",") if m.strip()]

    if len(target_names) != len(model_uris):
        raise ValueError(
            f"Mismatch between targets ({len(target_names)}) "
            f"and model URIs ({len(model_uris)})"
        )

    # Parse destination prefix for the MME models
    dest_bucket, dest_prefix = _parse_s3_uri(models_prefix_uri)
    # Ensure trailing slash
    dest_prefix = dest_prefix.rstrip("/") + "/"

    copied_models = []

    # Copy each model artifact into the MME prefix
    for tgt, src_uri in zip(target_names, model_uris):
        src_bucket, src_key = _parse_s3_uri(src_uri)
        # Use a stable filename per target
        dest_key = f"{dest_prefix}{tgt}.tar.gz"

        print(f"Copying model for target '{tgt}' from {src_uri} to s3://{dest_bucket}/{dest_key}")

        copy_source = {"Bucket": src_bucket, "Key": src_key}
        s3.copy_object(Bucket=dest_bucket, Key=dest_key, CopySource=copy_source)

        copied_models.append(
            {
                "target": tgt,
                "source": src_uri,
                "destination": f"s3://{dest_bucket}/{dest_key}",
            }
        )

    # Create or update the SageMaker multi-model endpoint
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{endpoint_name}-model-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    # Multi-model container definition
    container = {
        "Image": image_uri,
        "Mode": "MultiModel",
        "ModelDataUrl": models_prefix_uri.rstrip("/"),  # prefix only
        "Environment": {},
    }

    print(f"Creating model '{model_name}' with ModelDataUrl={container['ModelDataUrl']}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container,
        ExecutionRoleArn=exec_role_arn,
    )

    print(
        f"Creating endpoint config '{endpoint_config_name}' "
        f"for endpoint '{endpoint_name}' (type={instance_type}, count={initial_instance_count})"
    )
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # Check if endpoint exists
    endpoint_exists = False
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        print(f"Endpoint '{endpoint_name}' exists; will update.")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ValidationException" or "Could not find endpoint" in str(e):
            print(f"Endpoint '{endpoint_name}' does not exist; will create.")
            endpoint_exists = False
        else:
            raise

    if endpoint_exists:
        resp = sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "updated"
        print(f"Update endpoint response: {json.dumps(resp, default=str)}")
    else:
        resp = sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "created"
        print(f"Create endpoint response: {json.dumps(resp, default=str)}")

    # (Optional) wait for InService if you want, but usually you don't block here in a pipeline
    # waiter = sm.get_waiter("endpoint_in_service")
    # try:
    #     waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 60, "MaxAttempts": 60})
    #     endpoint_status = "InService"
    # except WaiterError:
    #     endpoint_status = "CreatingOrUpdating"

    result = {
        "status": f"endpoint_{action}",
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "models_prefix": models_prefix_uri,
        "copied_models": copied_models,
    }

    print("Result:", json.dumps(result))
    # LambdaStep expects a JSON-serializable object; the 'status' field
    # is wired in your LambdaStep outputs as a String.
    return result
