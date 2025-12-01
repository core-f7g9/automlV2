import json
import boto3
import os
import tarfile
import urllib.parse

s3 = boto3.client("s3")
sm = boto3.client("sagemaker")

"""
Expected Lambda Inputs:
{
    "EndpointName": "client1-mme",
    "InstanceType": "ml.m5.large",
    "InitialInstanceCount": "1",
    "ModelsPrefix": "s3://bucket/mlops/mme/client1/models/",
    "TargetNamesCSV": "DepartmentCode,AccountCode,...",
    "TargetImagesCSV": "<xgb_image>,<xgb_image>,<xgb_image>,<xgb_image>",
    "TargetModelDatasCSV": "s3://bucket/.../model.tar.gz,s3://bucket/.../model.tar.gz,..."
}
"""

def parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 URI: {uri}")
    parts = uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def copy_model_artifact(source_uri, dest_prefix, model_name):
    """
    Takes a model.tar.gz produced by SageMaker Training and uploads it to:

        <dest_prefix>/<model_name>/model.tar.gz

    required by MultiModel.
    """
    src_bucket, src_key = parse_s3_uri(source_uri)
    dst_bucket, dst_prefix = parse_s3_uri(dest_prefix)

    dst_key = f"{dst_prefix.rstrip('/')}/{model_name}/model.tar.gz"

    s3.copy_object(
        Bucket=dst_bucket,
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Key=dst_key
    )

    return f"s3://{dst_bucket}/{dst_key}"


def handler(event, context):
    print("EVENT:", json.dumps(event))

    endpoint_name  = event["EndpointName"]
    instance_type  = event["InstanceType"]
    initial_count  = int(event["InitialInstanceCount"])
    models_prefix  = event["ModelsPrefix"]

    target_names   = event["TargetNamesCSV"].split(",")
    images_csv     = event["TargetImagesCSV"].split(",")
    model_datas    = event["TargetModelDatasCSV"].split(",")

    # ----------------------------------------
    # 1. Copy models to MME prefix
    # ----------------------------------------
    copied_paths = []
    for tgt, src in zip(target_names, model_datas):
        print(f"Copying model for {tgt}: {src}")
        out_path = copy_model_artifact(src, models_prefix, tgt)
        copied_paths.append(out_path)

    # ----------------------------------------
    # 2. Create a Multi-Model container definition
    # ----------------------------------------
    # All XGBoost models share the same container
    model_image = images_csv[0]

    model_name = f"{endpoint_name}-mme-model"
    model_data_prefix_bucket, model_data_prefix_key = parse_s3_uri(models_prefix)

    container_def = {
        "Image": model_image,
        "Mode": "MultiModel",
        "ModelDataSource": {
            "S3DataSource": {
                "S3Uri": models_prefix,
                "S3DataType": "S3Prefix"
            }
        }
    }

    # ----------------------------------------
    # 3. Create / Update SageMaker Model
    # ----------------------------------------
    try:
        print(f"Creating model {model_name} ...")
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=os.environ.get("EXEC_ROLE_ARN"),
            PrimaryContainer=container_def
        )
    except sm.exceptions.ResourceInUse:
        print("Model already exists, updating not required")

    # ----------------------------------------
    # 4. Create / Update Endpoint Config
    # ----------------------------------------
    endpoint_config_name = f"{endpoint_name}-config"

    try:
        print(f"Creating endpoint config {endpoint_config_name} ...")
        sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllModels",
                    "ModelName": model_name,
                    "InitialInstanceCount": initial_count,
                    "InstanceType": instance_type,
                    "ServerlessConfig": {}
                }
            ]
        )
    except sm.exceptions.ResourceInUse:
        print("Endpoint config exists, replacing...")
        sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllModels",
                    "ModelName": model_name,
                    "InitialInstanceCount": initial_count,
                    "InstanceType": instance_type,
                }
            ]
        )

    # ----------------------------------------
    # 5. Create or Update Endpoint
    # ----------------------------------------
    try:
        print(f"Creating endpoint {endpoint_name} ...")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    except sm.exceptions.ResourceInUse:
        print(f"Updating endpoint {endpoint_name} ...")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )

    return {
        "status": "SUCCESS",
        "endpoint": endpoint_name,
        "models_copied": copied_paths
    }
