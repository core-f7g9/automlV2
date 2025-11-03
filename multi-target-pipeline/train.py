# ==========================================================
# Cell 3: Build & run a four-target Autopilot V1 pipeline
#          and deploy all 4 on ONE instance via MME
# ==========================================================
import boto3, sagemaker, time, json, os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel

# Autopilot V1
from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.automl_step import AutoMLStep

# Lambda deployment
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep

p_sess = PipelineSession()
region = boto3.Session().region_name

# --------- pipeline parameters ----------
bucket_param       = ParameterString("Bucket",       default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV",   default_value=INPUT_S3CSV)
val_frac_param     = ParameterFloat( "ValFrac",      default_value=0.2)
seed_param         = ParameterInteger("RandomSeed",  default_value=42)

# Deployment params
DeployAfterRegister   = ParameterBoolean(name="DeployAfterRegister", default_value=True)
EndpointNameParam     = ParameterString(name="EndpointName", default_value=f"{PROJECT_NAME}-codes-mme")
InstanceTypeParam     = ParameterString(name="InstanceType", default_value="ml.m5.large")
InitialInstanceCount  = ParameterInteger(name="InitialInstanceCount", default_value=1)
DataCaptureS3Param    = ParameterString(name="DataCaptureS3Uri", default_value=f"s3://{BUCKET}/{OUTPUT_PREFIX}/data-capture/")
CapturePercentParam   = ParameterInteger(name="CapturePercent", default_value=100)

# literals (keeps Autopilot happy at compile time)
PROBLEM_TYPE = "MulticlassClassification"
OBJECTIVE    = "Accuracy"

# --------- Step 1: Processing (split once) ----------
img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
script_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

# Use DepartmentCode for stratification; all four targets exist in CSV
split_step = ProcessingStep(
    name="SplitTrainValidation",
    processor=script_processor,
    inputs=[ProcessingInput(source=input_s3_csv_param, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train",      source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
    ],
    code="sql_to_s3_and_split.py",
    job_arguments=[
        "--target_col",   "DepartmentCode",
        "--val_frac",     val_frac_param.to_string(),
        "--random_seed",  seed_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir",   "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

train_prefix_s3 = split_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
val_prefix_s3   = split_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri

# --------- Per-target branch builder: Autopilot + Register, expose image & model_data ---------
def build_target_branch(target_name: str):
    auto_inputs = [
        AutoMLInput(
            inputs=train_prefix_s3,
            channel_type="training",
            content_type="text/csv;header=present",
            target_attribute_name=target_name
        ),
        AutoMLInput(
            inputs=val_prefix_s3,
            channel_type="validation",
            content_type="text/csv;header=present",
            target_attribute_name=target_name
        ),
    ]
    auto_ml_job_config = {
        "CandidateGenerationConfig": {
            "FeatureSpecificationS3Uri": FEATURE_SPEC_S3  # << whitelist the 3 inputs
        }
    }
    automl = AutoML(
        role=role_arn,
        sagemaker_session=p_sess,
        target_attribute_name=target_name,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/autopilot-output/{target_name}/",
        problem_type=PROBLEM_TYPE,
        job_objective={"MetricName": OBJECTIVE},
        max_candidates=5,
        mode="ENSEMBLING",
        max_runtime_per_training_job_in_seconds=1800,
        total_job_runtime_in_seconds=6*3600,
        auto_ml_job_config=auto_ml_job_config
    )
    step_args = automl.fit(inputs=auto_inputs)
    automl_step = AutoMLStep(name=f"RunAutopilotV1_{target_name}", step_args=step_args)

    best_image = automl_step.properties.BestCandidate.InferenceContainers[0].Image
    best_data  = automl_step.properties.BestCandidate.InferenceContainers[0].ModelDataUrl

    # Register to its own package group
    model_to_register = Model(
        image_uri=best_image,
        model_data=best_data,
        role=role_arn,
        sagemaker_session=p_sess,
    )
    register_step = RegisterModel(
        name=f"RegisterBestModel_{target_name}",
        model=model_to_register,
        content_types=["text/csv","application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large","ml.c5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{PROJECT_NAME}-pkg-group-{target_name}",
        approval_status="Approved",
        description=f"Best model for target {target_name} (whitelisted features: {INPUT_FEATURES})",
    )
    return automl_step, register_step, best_image, best_data

branches = []
best_images = {}
best_datas  = {}
for tgt in TARGET_COLS:
    a_step, r_step, img_uri, data_uri = build_target_branch(tgt)
    branches += [a_step, r_step]
    best_images[tgt] = img_uri
    best_datas[tgt]  = data_uri

# --------- Lambda for MME deployment (all four on one instance) ----------
# This Lambda:
# 1) Validates all 4 images are identical (MME requires single container)
# 2) Copies each model.tar.gz to a shared S3 prefix: s3://.../mme/{EndpointName}/models/{TargetName}.tar.gz
# 3) Creates/updates a SINGLE-variant endpoint (one instance) with Mode="MultiModel"
# 4) Enables data capture
deploy_lambda_src = r"""
import boto3, os, time, json
from urllib.parse import urlparse

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

def _s3copy(src_uri, dst_uri):
    def p(u):
        up = urlparse(u)
        return up.netloc, up.path.lstrip("/")
    sb, sk = p(src_uri)
    db, dk = p(dst_uri)
    s3.copy_object(Bucket=db, Key=dk, CopySource={"Bucket": sb, "Key": sk})

def handler(event, context):
    endpoint   = event["EndpointName"]
    inst_type  = event["InstanceType"]
    init_count = int(event["InitialInstanceCount"])
    exec_role  = os.environ["EXEC_ROLE_ARN"]
    data_cap_s3 = event["DataCaptureS3Uri"]
    cap_pct     = int(event.get("CapturePercent", 100))
    mme_prefix  = event["ModelsPrefix"]   # s3://bucket/prefix/mme/{endpoint}/models/

    # Inputs: per-target image + model data
    targets = event["Targets"]  # list of dicts: { "name": "...", "image": "...", "model_data": "s3://.../model.tar.gz" }

    # 1) Validate single image across all targets
    images = { t["image"] for t in targets }
    if len(images) != 1:
        raise ValueError(f"MME requires a single container image. Found: {list(images)}. "
                         f"Use separate endpoints or retrain to a common framework.")

    image = list(images)[0]

    # 2) Copy each model.tar.gz to shared prefix with deterministic key
    #    e.g., s3://bucket/prefix/mme/<endpoint>/models/DepartmentCode.tar.gz
    up = urlparse(mme_prefix)
    dst_bucket, dst_key_prefix = up.netloc, up.path.lstrip("/")
    if not dst_key_prefix.endswith("/"):
        dst_key_prefix += "/"

    for t in targets:
        dst_key = f"{dst_key_prefix}{t['name']}.tar.gz"
        dst_uri = f"s3://{dst_bucket}/{dst_key}"
        _s3copy(t["model_data"], dst_uri)

    # 3) Create (or update) a single Multi-Model container Model and EndpointConfig
    stamp = str(int(time.time()))
    model_name = f"{endpoint}-mme-model-{stamp}"
    cfg_name   = f"{endpoint}-mme-cfg-{stamp}"

    # Create the Multi-Model container model (Mode="MultiModel"), ModelDataUrl points to prefix (not a single tar)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image,
            "Mode": "MultiModel",
            "ModelDataUrl": mme_prefix  # S3 prefix where our per-target .tar.gz files live
        },
        ExecutionRoleArn=exec_role
    )

    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "AllCodes",
            "ModelName": model_name,
            "InstanceType": inst_type,
            "InitialInstanceCount": init_count
        }],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": cap_pct,
            "DestinationS3Uri": data_cap_s3,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": ["text/csv"],
                "JsonContentTypes": ["application/json"]
            }
        }
    )

    # Upsert the endpoint
    try:
        sm.describe_endpoint(EndpointName=endpoint)
        sm.update_endpoint(EndpointName=endpoint, EndpointConfigName=cfg_name)
        action = "updated"
    except sm.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e) or "NotFound" in str(e):
            sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=cfg_name)
            action = "created"
        else:
            raise

    return {
        "status": "OK",
        "action": action,
        "endpoint": endpoint,
        "config": cfg_name,
        "model": model_name,
        "mme_prefix": mme_prefix
    }
"""

with open("deploy_mme_from_autopilot.py", "w") as f:
    f.write(deploy_lambda_src)

deploy_lambda_name = f"{PROJECT_NAME}-deploy-mme-{int(time.time())}"
deploy_lam = Lambda(
    function_name=deploy_lambda_name,
    execution_role_arn=role_arn,  # role must allow SageMaker, S3 (Get/Put/Copy), CloudWatch Logs
    script="deploy_mme_from_autopilot.py",
    handler="deploy_mme_from_autopilot.handler",
    timeout=600,
    memory_size=512,
    environment={"Variables": {"EXEC_ROLE_ARN": role_arn}},
)

# Build the Lambda input payload fields from step properties
targets_payload = {}
for tgt in TARGET_COLS:
    targets_payload[tgt] = {
        "image": str(best_images[tgt]),
        "model_data": str(best_datas[tgt]),
        "name": tgt
    }

# S3 prefix to host the MME model files (one tar per target)
MME_MODELS_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/{str(int(time.time()))}/models/"

deploy_step = LambdaStep(
    name="DeployAllFourOnOneInstance_MME",
    lambda_func=deploy_lam,
    inputs={
        "EndpointName":    EndpointNameParam,
        "InstanceType":    InstanceTypeParam,
        "InitialInstanceCount": InitialInstanceCount,
        "DataCaptureS3Uri": DataCaptureS3Param,
        "CapturePercent":  CapturePercentParam,
        "ModelsPrefix":    MME_MODELS_PREFIX,
        # flatten the targets list for LambdaStep (it needs JSON-serializable primitives)
        "Targets": [targets_payload[t] for t in TARGET_COLS],
    },
    outputs=[LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)]
)

should_deploy = ConditionEquals(left=DeployAfterRegister, right=True)
deploy_condition_step = ConditionStep(
    name="MaybeDeployAllFourOnOneInstance_MME",
    conditions=[should_deploy],
    if_steps=[deploy_step],
    else_steps=[],
)

# --------- Assemble pipeline ----------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-4targets-mme",
    parameters=[
        bucket_param, input_s3_csv_param, val_frac_param, seed_param,
        DeployAfterRegister, EndpointNameParam, InstanceTypeParam, InitialInstanceCount,
        DataCaptureS3Param, CapturePercentParam
    ],
    steps=[split_step] + branches + [deploy_condition_step],
    sagemaker_session=p_sess,
)

pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.arn
