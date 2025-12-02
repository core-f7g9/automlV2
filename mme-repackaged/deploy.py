import json
import boto3
from urllib.parse import urlparse

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

def parse_s3(uri):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")

def handler(event, context):
    endpoint_name = event["EndpointName"]
    models_prefix = event["ModelsPrefix"]
    client_name   = event["ClientName"]
    target_cols   = event["TargetsCSV"].split(",")
    instance_type = event["InstanceType"]
    instance_count = int(event["InstanceCount"])

    dest_bucket, dest_prefix = parse_s3(models_prefix)
    if not dest_prefix.endswith("/"):
        dest_prefix += "/"

    # built-in xgboost container
    region = boto3.Session().region_name
    xgb_image = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.5-1"

    # Copy model tar files
    for tgt in target_cols:
        pkg_group = f"{client_name}-{tgt}-models"
        resp = sm.list_model_packages(
            ModelPackageGroupName=pkg_group,
            SortBy="CreationTime",
            SortOrder="Descending"
        )
        pkg_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]

        desc = sm.describe_model_package(ModelPackageName=pkg_arn)
        model_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

        src_bucket, src_key = parse_s3(model_uri)
        dst_key = f"{dest_prefix}{tgt}.tar.gz"

        s3.copy_object(
            Bucket=dest_bucket,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key}
        )

    model_name = f"{endpoint_name}-mme-model"
    try:
        sm.describe_model(ModelName=model_name)
    except:
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=event["ExecRoleArn"],
            PrimaryContainer={
                "Image": xgb_image,
                "ModelDataUrl": models_prefix,
                "Mode": "MultiModel",
            }
        )

    config_name = f"{endpoint_name}-config"

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "ModelName": model_name,
                "VariantName": "AllTraffic",
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
            }
        ]
    )

    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

    return {"status": "OK"}
