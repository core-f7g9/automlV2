# ==========================================================
# Cell 3: Build & run 4-target Autopilot V1 pipeline
#          Deploy all on ONE instance via MME
# ==========================================================
import boto3, sagemaker, time, json
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

# --------- Step 1: Processing (split once, write per-target subsets) ----------
img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
script_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

# Dynamically define outputs for each target (train & validation)
processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.append(ProcessingOutput(output_name=f"train_{tgt}",      source=f"/opt/ml/processing/output/{tgt}/train"))
    processing_outputs.append(ProcessingOutput(output_name=f"validation_{tgt}", source=f"/opt/ml/processing/output/{tgt}/validation"))

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=script_processor,
    inputs=[ProcessingInput(source=input_s3_csv_param, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--stratify_target", TARGET_COLS[0],  # stratify on first target
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac_param.to_string(),
        "--random_seed", seed_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

# --------- Helper: per-target Autopilot + Register ----------
def build_target_branch(target_name: str):
    train_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"train_{target_name}"].S3Output.S3Uri
    val_s3   = split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{target_name}"].S3Output.S3Uri

    auto_inputs = [
        AutoMLInput(
            inputs=train_s3,
            channel_type="training",
            content_type="text/csv;header=present",
            target_attribute_name=target_name
        ),
        AutoMLInput(
            inputs=val_s3,
            channel_type="validation",
            content_type="text/csv;header=present",
            target_attribute_name=target_name
        ),
    ]

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
        total_job_runtime_in_seconds=6*3600
        # NOTE: No SDK whitelist flags needed—CSV already has only inputs + target
    )

    step_args = automl.fit(inputs=auto_inputs)
    automl_step = AutoMLStep(name=f"RunAutopilotV1_{target_name}", step_args=step_args)

    best_image = automl_step.properties.BestCandidate.InferenceContainers[0].Image
    best_data  = automl_step.properties.BestCandidate.InferenceContainers[0].ModelDataUrl

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
        description=f"Best model for target {target_name} (columns: {INPUT_FEATURES + [target_name]})",
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

# --------- Lambda for MME deploy (same as before, validates common image) ----------
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

    targets = event["Targets"]  # list of dicts: { "name": "...", "image": "...", "model_data": "s3://.../model.tar.gz" }

    images = { t["image"] for t in targets }
    if len(images) != 1:
        raise ValueError(f"MME requires a single container image. Found: {list(images)}.")

    image = list(images)[0]

    from urllib.parse import urlparse
    up = urlparse(mme_prefix)
    dst_bucket, dst_key_prefix = up.netloc, up.path.lstrip("/")
    if not dst_key_prefix.endswith("/"):
        dst_key_prefix += "/"

    for t in targets:
        dst_key = f"{dst_key_prefix}{t['name']}.tar.gz"
        dst_uri = f"s3://{dst_bucket}/{dst_key}"
        _s3copy(t["model_data"], dst_uri)

    stamp = str(int(time.time()))
    model_name = f"{endpoint}-mme-model-{stamp}"
    cfg_name   = f"{endpoint}-mme-cfg-{stamp}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image,
            "Mode": "MultiModel",
            "ModelDataUrl": mme_prefix
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
    execution_role_arn=role_arn,
    script="deploy_mme_from_autopilot.py",
    handler="deploy_mme_from_autopilot.handler",
    timeout=600,
    memory_size=512,
    environment={"Variables": {"EXEC_ROLE_ARN": role_arn}},
)

# Build Lambda input payload from step properties
targets_payload = []
for tgt in TARGET_COLS:
    targets_payload.append({
        "name": tgt,
        "image": str(best_images[tgt]),
        "model_data": str(best_datas[tgt]),
    })

MME_MODELS_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/{str(int(time.time()))}/models/"

deploy_step = LambdaStep(
    name="DeployAllOnOneInstance_MME",
    lambda_func=deploy_lam,
    inputs={
        "EndpointName":    EndpointNameParam,
        "InstanceType":    InstanceTypeParam,
        "InitialInstanceCount": InitialInstanceCount,
        "DataCaptureS3Uri": DataCaptureS3Param,
        "CapturePercent":  CapturePercentParam,
        "ModelsPrefix":    MME_MODELS_PREFIX,
        "Targets":         targets_payload,
    },
    outputs=[LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)]
)

should_deploy = ConditionEquals(left=DeployAfterRegister, right=True)
deploy_condition_step = ConditionStep(
    name="MaybeDeployAllOnOneInstance_MME",
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
