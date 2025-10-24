# ============================================
# Cell 3: Build & run the SageMaker Pipeline (Autopilot V1, folder outputs)
# ============================================
import json, boto3, sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel

# Autopilot V1
from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.automl_step import AutoMLStep

# Deployment
from sagemaker.workflow.parameters import ParameterBoolean, ParameterString, ParameterInteger

DeployAfterRegister   = ParameterBoolean(name="DeployAfterRegister", default_value=False)
EndpointNameParam     = ParameterString(name="EndpointName", default_value=f"{PROJECT_NAME}-endpoint")
InstanceTypeParam     = ParameterString(name="InstanceType", default_value="ml.m5.large")
InitialInstanceCount  = ParameterInteger(name="InitialInstanceCount", default_value=1)

p_sess = PipelineSession()
region = boto3.Session().region_name

# --------- pipeline parameters ----------
bucket_param       = ParameterString("Bucket",       default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV",   default_value=INPUT_S3CSV)
target_col_param   = ParameterString("TargetCol",    default_value=TARGET_COL)
val_frac_param     = ParameterFloat( "ValFrac",      default_value=0.2)
seed_param         = ParameterInteger("RandomSeed",  default_value=42)

# literals (keeps Autopilot happy at compile time)
PROBLEM_TYPE = "MulticlassClassification"
OBJECTIVE    = "Accuracy"

# --------- Step 1: Processing (split) ----------
img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
script_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,                 # plain string from Cell 1
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

split_step = ProcessingStep(
    name="SplitTrainValidation",
    processor=script_processor,
    inputs=[
        # Mount the exact S3 object under /opt/ml/processing/input/
        ProcessingInput(
            source=input_s3_csv_param,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train",      source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
    ],
    code="sql_to_s3_and_split.py",  # reads mounted file
    job_arguments=[
        # NOTE: no --input_s3_csv; script reads mounted file
        "--target_col",   target_col_param,
        "--val_frac",     val_frac_param.to_string(),
        "--random_seed",  seed_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir",   "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

# Use the S3 prefixes of the outputs
train_prefix_s3 = split_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
val_prefix_s3   = split_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri

# --------- Step 2: Autopilot V1 (now with validation channel) ----------
auto_inputs = [
    AutoMLInput(
        inputs=train_prefix_s3,
        channel_type="training",
        content_type="text/csv;header=present",
        target_attribute_name=target_col_param
    ),
    AutoMLInput(
        inputs=val_prefix_s3,
        channel_type="validation",
        content_type="text/csv;header=present",
        target_attribute_name=target_col_param
    ),
]

automl = AutoML(
    role=role_arn,
    sagemaker_session=p_sess,
    target_attribute_name=target_col_param,
    output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/autopilot-output/",
    problem_type=PROBLEM_TYPE,
    job_objective={"MetricName": OBJECTIVE},
    max_candidates=1,
    mode="ENSEMBLING",
    max_runtime_per_training_job_in_seconds=600,
    total_job_runtime_in_seconds=1800
)

# Autopilot V1: fit() returns step_args for AutoMLStep
step_args = automl.fit(inputs=auto_inputs)

automl_step = AutoMLStep(
    name="RunAutopilotV1",
    step_args=step_args
)

# Best candidate artifacts from the AutoML step properties (V1)
best_image = automl_step.properties.BestCandidate.InferenceContainers[0].Image
best_data  = automl_step.properties.BestCandidate.InferenceContainers[0].ModelDataUrl

# --------- Step 3: Register best model to Model Registry ----------
model_to_register = Model(
    image_uri=best_image,
    model_data=best_data,
    role=role_arn,
    sagemaker_session=p_sess,
)

register_step = RegisterModel(
    name="RegisterBestModel",
    model=model_to_register,
    content_types=["text/csv","application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large","ml.c5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=f"{PROJECT_NAME}-pkg-group",
    approval_status="Approved",
    description="Best model from Autopilot V1",
)

# --------- Deployment (unchanged) ----------
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
import time

deploy_lambda_src = r"""
import boto3, os, time
sm = boto3.client("sagemaker")

def handler(event, context):
    pkg_arn    = event["ModelPackageArn"]
    endpoint   = event["EndpointName"]
    inst_type  = event["InstanceType"]
    init_count = int(event["InitialInstanceCount"])
    exec_role  = os.environ["EXEC_ROLE_ARN"]

    stamp = str(int(time.time()))
    model_name = f"{endpoint}-model-{stamp}"
    cfg_name   = f"{endpoint}-cfg-{stamp}"

    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": pkg_arn},
            ExecutionRoleArn=exec_role
        )
    except sm.exceptions.ClientError as e:
        if "AlreadyExists" not in str(e):
            raise

    try:
        sm.create_endpoint_config(
            EndpointConfigName=cfg_name,
            ProductionVariants=[{
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": inst_type,
                "InitialInstanceCount": init_count
            }]
        )
    except sm.exceptions.ClientError as e:
        if "AlreadyExists" not in str(e):
            raise

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

    return {"status": "OK", "action": action, "endpoint": endpoint, "config": cfg_name, "model": model_name}
"""

with open("deploy_from_registry.py", "w") as f:
    f.write(deploy_lambda_src)

deploy_lambda_name = f"{PROJECT_NAME}-deploy-{int(time.time())}"
deploy_lam = Lambda(
    function_name=deploy_lambda_name,
    execution_role_arn=role_arn,   # keep as-is if your role already works for Lambda+SageMaker
    script="deploy_from_registry.py",
    handler="deploy_from_registry.handler",
    timeout=300,
    memory_size=256,
    environment={"Variables": {"EXEC_ROLE_ARN": role_arn}},
)

model_package_arn_prop = register_step.properties.ModelPackageArn

deploy_step = LambdaStep(
    name="DeployEndpointDedicated",
    lambda_func=deploy_lam,
    inputs={
        "ModelPackageArn": model_package_arn_prop,
        "EndpointName":    EndpointNameParam,
        "InstanceType":    InstanceTypeParam,
        "InitialInstanceCount": InitialInstanceCount,
    },
    outputs=[LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)]
)

should_deploy = ConditionEquals(left=DeployAfterRegister, right=True)
deploy_condition_step = ConditionStep(
    name="MaybeDeployDedicated",
    conditions=[should_deploy],
    if_steps=[deploy_step],
    else_steps=[],
)

# --------- Build & start the pipeline ----------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[
        bucket_param, input_s3_csv_param, target_col_param, val_frac_param, seed_param,
        DeployAfterRegister, EndpointNameParam, InstanceTypeParam, InitialInstanceCount
    ],
    steps=[split_step, automl_step, register_step, deploy_condition_step],
    sagemaker_session=p_sess,
)

pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.arn
