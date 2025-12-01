# =========================
# Pipeline B: Setup variables
# =========================

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

region = boto3.Session().region_name
sm = boto3.client("sagemaker")
sm_sess = sagemaker.Session()
p_sess = PipelineSession()
role_arn = sagemaker.get_execution_role()

CLIENT_NAME = "client1"
TARGET_COLS = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]

BUCKET = sm_sess.default_bucket()
MME_PREFIX = f"s3://{BUCKET}/mme/{CLIENT_NAME}/models/"
ENDPOINT_NAME = f"{CLIENT_NAME}-mme-endpoint"

INSTANCE_TYPE = "ml.m5.large"
INSTANCE_COUNT = 1

print("Bucket:", BUCKET)
print("MME Prefix:", MME_PREFIX)
print("Endpoint:", ENDPOINT_NAME)

# ============================================================
# Cell 2: Lambda script for MME deployment (FINAL WORKING VERSION)
# ============================================================
import os, textwrap

SKLEARN_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

lambda_script = textwrap.dedent("""
import json
import boto3
import os
from urllib.parse import urlparse

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

ROLE_ARN = "__ROLE_ARN__"
SKLEARN_IMAGE = "__SKLEARN_IMAGE__"

def _parse_s3(uri):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def handler(event, context):
    client_name     = event["ClientName"]
    targets_csv     = event["Targets"]
    mme_prefix      = event["MMEPrefix"]
    endpoint_name   = event["EndpointName"]
    instance_type   = event["InstanceType"]
    instance_count  = int(event["InstanceCount"])

    targets = [t.strip() for t in targets_csv.split(",") if t.strip()]

    dest_bucket, dest_prefix = _parse_s3(mme_prefix)
    if not dest_prefix.endswith("/"):
        dest_prefix += "/"

    deployed_models = {}

    # ---- Step 1: Find latest Approved model for each target ----
    for tgt in targets:
        group = "{}-{}-models".format(client_name, tgt)

        resp = sm.list_model_packages(
            ModelPackageGroupName=group,
            SortBy="CreationTime",
            SortOrder="Descending",
        )

        if len(resp["ModelPackageSummaryList"]) == 0:
            raise ValueError("No registered models for {}".format(group))

        pkg = resp["ModelPackageSummaryList"][0]
        pkg_name = pkg["ModelPackageArn"]

        details = sm.describe_model_package(ModelPackageName=pkg_name)
        data_uri = details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

        deployed_models[tgt] = data_uri

        # ---- Copy model.tar.gz -> MME prefix ----
        src_bucket, src_key = _parse_s3(data_uri)
        dest_key = "{}{}.tar.gz".format(dest_prefix, tgt)

        s3.copy_object(
            Bucket=dest_bucket,
            Key=dest_key,
            CopySource={
                "Bucket": src_bucket,
                "Key": src_key
            },
        )

    # ---- Step 2: Create Multi-Model container definition ----
    model_name = "{}-mme-model".format(endpoint_name)

    container_def = {
        "Image": SKLEARN_IMAGE,
        "ModelDataUrl": mme_prefix,
        "Mode": "MultiModel",
    }

    # ---- Step 3: Create or reuse SageMaker Model ----
    try:
        sm.describe_model(ModelName=model_name)
        print("Reusing existing model {}".format(model_name))
    except:
        print("Creating model {}".format(model_name))
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=ROLE_ARN,
            Containers=[container_def],
        )

    # ---- Step 4: Create fresh endpoint config ----
    import time
    config_name = "{}-config-{}".format(endpoint_name, int(time.time()))

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "ModelName": model_name,
                "VariantName": "AllTraffic",
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
            }
        ],
    )

    # ---- Step 5: Create or Update endpoint ----
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print("Updating endpoint {}".format(endpoint_name))
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except:
        print("Creating endpoint {}".format(endpoint_name))
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

    return {
        "status": "OK",
        "endpoint": endpoint_name,
        "mme_prefix": mme_prefix,
        "deployed_models": deployed_models
    }
""")

# Inject role ARN and image safely
lambda_script = lambda_script.replace("__ROLE_ARN__", role_arn)
lambda_script = lambda_script.replace("__SKLEARN_IMAGE__", SKLEARN_IMAGE)

with open("deploy_mme_lambda.py", "w") as f:
    f.write(lambda_script)

print("Wrote deploy_mme_lambda.py (FINAL WORKING VERSION)")

# ============================================================
# Cell 3: Pipeline B definition (Deployment Pipeline)
# ============================================================
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum

lambda_deploy = Lambda(
    function_name=f"{CLIENT_NAME}-mme-deploy-lambda",
    execution_role_arn=role_arn,
    script="deploy_mme_lambda.py",
    handler="deploy_mme_lambda.handler",
    timeout=900,
    memory_size=512,
)

client_param = ParameterString("ClientName", default_value=CLIENT_NAME)
targets_param = ParameterString("Targets", default_value=",".join(TARGET_COLS))
mme_prefix_param = ParameterString("MMEPrefix", default_value=MME_PREFIX)
endpoint_param = ParameterString("EndpointName", default_value=ENDPOINT_NAME)
instance_type_param = ParameterString("InstanceType", default_value=INSTANCE_TYPE)
instance_count_param = ParameterInteger("InstanceCount", default_value=INSTANCE_COUNT)

deploy_step = LambdaStep(
    name="DeployMMEModels",
    lambda_func=lambda_deploy,
    inputs={
        "ClientName": client_param,
        "Targets": targets_param,
        "MMEPrefix": mme_prefix_param,
        "EndpointName": endpoint_param,
        "InstanceType": instance_type_param,
        "InstanceCount": instance_count_param,
    },
    outputs=[
        LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)
    ],
)

pipeline_b = Pipeline(
    name=f"{CLIENT_NAME}-mme-deploy-pipeline",
    parameters=[
        client_param,
        targets_param,
        mme_prefix_param,
        endpoint_param,
        instance_type_param,
        instance_count_param,
    ],
    steps=[deploy_step],
    sagemaker_session=p_sess,
)

pipeline_b.upsert(role_arn=role_arn)

# -------------------------------
# Add the required 20 second wait
# -------------------------------
import time
print("Waiting 20 seconds for Lambda to finish creating...")
time.sleep(20)

execution = pipeline_b.start()
print("Pipeline B execution started:", execution.arn)

