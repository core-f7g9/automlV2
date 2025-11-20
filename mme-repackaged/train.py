# multi-target-pipeline/train.py
# ==========================================================
# Cell 3: Per-target Autopilot V1 + MME deployment (1 instance)
# ==========================================================
import time, json, os
import boto3, sagemaker

# ---- Config (edit these in your notebook cell 1, or override via env vars) ----
region = boto3.Session().region_name

def _default_role():
    try:
        return sagemaker.get_execution_role()
    except Exception:
        return os.getenv("SAGEMAKER_ROLE_ARN") or os.getenv("ROLE_ARN")

def _default_bucket():
    try:
        return sagemaker.Session().default_bucket()
    except Exception:
        return os.getenv("SAGEMAKER_DEFAULT_BUCKET")

CLIENT_NAME = globals().get("CLIENT_NAME", os.getenv("CLIENT_NAME", "client1"))
PROJECT_NAME = globals().get("PROJECT_NAME", os.getenv("PROJECT_NAME", f"{CLIENT_NAME}-autopilot-v1"))
OUTPUT_PREFIX = globals().get("OUTPUT_PREFIX", os.getenv("OUTPUT_PREFIX", "mlops"))
BUCKET = globals().get("BUCKET", os.getenv("BUCKET", _default_bucket()))
INPUT_S3CSV = globals().get("INPUT_S3CSV", os.getenv("INPUT_S3CSV", f"s3://{BUCKET}/input/data.csv"))
TARGET_COLS = globals().get("TARGET_COLS", ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"])
INPUT_FEATURES = globals().get("INPUT_FEATURES", ["VendorName", "LineDescription", "ClubNumber"])
VAL_FRAC_DEFAULT = globals().get("VAL_FRAC_DEFAULT", 0.20)
MIN_SUPPORT_DEFAULT = globals().get("MIN_SUPPORT_DEFAULT", 5)
RARE_TRAIN_ONLY_DEFAULT = globals().get("RARE_TRAIN_ONLY_DEFAULT", True)
CLIENT_NAME = globals().get("CLIENT_NAME", CLIENT_NAME)
role_arn = globals().get("role_arn", os.getenv("role_arn") or _default_role())

if not BUCKET:
    raise RuntimeError("BUCKET is not set. Set BUCKET or run this in SageMaker Studio so default_bucket() is available.")
if not role_arn:
    raise RuntimeError("role_arn is not set. Set role_arn (or env ROLE_ARN/SAGEMAKER_ROLE_ARN) before running.")

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.functions import Join

from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.automl_step import AutoMLStep

from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep

p_sess = PipelineSession()

bucket_param       = ParameterString("Bucket",       default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV",   default_value=INPUT_S3CSV)
val_frac_param     = ParameterFloat( "ValFrac",      default_value=VAL_FRAC_DEFAULT)
seed_param         = ParameterInteger("RandomSeed",  default_value=42)
min_support_param  = ParameterInteger("MinSupport",  default_value=MIN_SUPPORT_DEFAULT)
rare_train_only_param = ParameterBoolean("RareTrainOnly", default_value=RARE_TRAIN_ONLY_DEFAULT)

DeployAfterRegister   = ParameterBoolean(name="DeployAfterRegister", default_value=True)
EndpointNameParam     = ParameterString(name="EndpointName", default_value=f"{PROJECT_NAME}-codes-mme")
InstanceTypeParam     = ParameterString(name="InstanceType", default_value="ml.m5.large")
InitialInstanceCount  = ParameterInteger(name="InitialInstanceCount", default_value=1)

PROBLEM_TYPE = "MulticlassClassification"
OBJECTIVE    = "Accuracy"

img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
split_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)
repack_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.append(
        ProcessingOutput(
            output_name=f"train_{tgt}",
            source=f"/opt/ml/processing/output/{tgt}/train"
        )
    )
    processing_outputs.append(
        ProcessingOutput(
            output_name=f"validation_{tgt}",
            source=f"/opt/ml/processing/output/{tgt}/validation"
        )
    )

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=split_processor,
    inputs=[ProcessingInput(source=input_s3_csv_param, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac",     val_frac_param.to_string(),
        "--random_seed",  seed_param.to_string(),
        "--min_support",  min_support_param.to_string(),
        "--rare_train_only", rare_train_only_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir",   "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

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
    )

    automl_step = AutoMLStep(
        name=f"RunAutopilotV1_{target_name}",
        step_args=automl.fit(inputs=auto_inputs)
    )

    best_image = automl_step.properties.BestCandidate.InferenceContainers[0].Image
    best_data  = automl_step.properties.BestCandidate.InferenceContainers[0].ModelDataUrl

    repack_step = ProcessingStep(
        name=f"RepackBestModel_{target_name}",
        processor=repack_processor,
        inputs=[
            ProcessingInput(
                source=best_data,
                destination="/opt/ml/processing/input/model"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="repacked_model",
                source="/opt/ml/processing/output/repacked_model"
            )
        ],
        code="repack_for_mme.py",
        job_arguments=[
            "--target_name", target_name,
            "--feature_list_csv", ",".join(INPUT_FEATURES),
            "--input_dir", "/opt/ml/processing/input/model",
            "--output_dir", "/opt/ml/processing/output/repacked_model",
        ],
    )

    repacked_prefix = repack_step.properties.ProcessingOutputConfig.Outputs["repacked_model"].S3Output.S3Uri
    repacked_model_s3 = Join(on="", values=[repacked_prefix, "/model.tar.gz"])

    model_to_register = Model(
        image_uri=best_image,
        model_data=repacked_model_s3,
        role=role_arn,
        sagemaker_session=p_sess,
    )

    register_step = RegisterModel(
        name=f"RegisterBestModel_{target_name}",
        model=model_to_register,
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.c5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{PROJECT_NAME}-pkg-group-{target_name}",
        approval_status="Approved",
        description=f"Best model for target {target_name} (cols: {INPUT_FEATURES + [target_name]})",
    )

    return automl_step, repack_step, register_step, best_image, repacked_model_s3

branches = []
best_images = {}
best_datas  = {}
for tgt in TARGET_COLS:
    a_step, repack_step, r_step, img_uri, data_uri = build_target_branch(tgt)
    branches += [a_step, repack_step, r_step]
    best_images[tgt] = img_uri
    best_datas[tgt]  = data_uri

deploy_lambda_src = r"""
import boto3, os, time
from urllib.parse import urlparse

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

def _s3copy(src_uri, dst_uri):
    up = urlparse(src_uri)
    sb, sk = up.netloc, up.path.lstrip("/")
    up2 = urlparse(dst_uri)
    db, dk = up2.netloc, up2.path.lstrip("/")
    s3.copy_object(Bucket=db, Key=dk, CopySource={"Bucket": sb, "Key": sk})

def handler(event, context):
    endpoint   = event["EndpointName"]
    inst_type  = event["InstanceType"]
    init_count = int(event["InitialInstanceCount"])
    exec_role  = os.environ["EXEC_ROLE_ARN"]
    mme_prefix = event["ModelsPrefix"]

    names  = [x for x in event["TargetNamesCSV"].split(",")  if x]
    images = [x for x in event["TargetImagesCSV"].split(",") if x]
    datas  = [x for x in event["TargetModelDatasCSV"].split(",") if x]

    if not (len(names) == len(images) == len(datas)):
        raise ValueError("Targets length mismatch among names/images/model_datas")

    if len(set(images)) != 1:
        raise ValueError(f"MME requires a single container image. Found: {list(set(images))}. Ensure all targets share the same AutoML inference container (try ENSEMBLING mode) before deploying.")

    image = images[0]

    up = urlparse(mme_prefix)
    dst_bucket, dst_key_prefix = up.netloc, up.path.lstrip("/")
    if not dst_key_prefix.endswith("/"):
        dst_key_prefix += "/"

    # Copy each repacked model to a stable key under the MME prefix
    for name, model_data in zip(names, datas):
        dst_key = f"{dst_key_prefix}{name}.tar.gz"
        dst_uri = f"s3://{dst_bucket}/{dst_key}"
        _s3copy(model_data, dst_uri)

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
        }]
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

deploy_lambda_name = f"{PROJECT_NAME}-deploy-mme"
deploy_lam = Lambda(
    function_name=deploy_lambda_name,
    execution_role_arn=role_arn,
    script="deploy_mme_from_autopilot.py",
    handler="deploy_mme_from_autopilot.handler",
    timeout=600,
    memory_size=512,
    environment={"Variables": {"EXEC_ROLE_ARN": role_arn}},
)

target_names_csv  = ",".join(TARGET_COLS)
target_images_csv = Join(on=",", values=[best_images[t].to_string() for t in TARGET_COLS])
target_datas_csv  = Join(on=",", values=[best_datas[t] for t in TARGET_COLS])

MME_MODELS_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/models/"

deploy_step = LambdaStep(
    name="DeployAllOnOneInstance_MME",
    lambda_func=deploy_lam,
    inputs={
        "EndpointName":          EndpointNameParam.to_string(),
        "InstanceType":          InstanceTypeParam.to_string(),
        "InitialInstanceCount":  InitialInstanceCount.to_string(),
        "ModelsPrefix":          MME_MODELS_PREFIX,
        "TargetNamesCSV":        target_names_csv,
        "TargetImagesCSV":       target_images_csv,
        "TargetModelDatasCSV":   target_datas_csv,
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

pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-4targets-mme",
    parameters=[
        bucket_param, input_s3_csv_param, val_frac_param, seed_param,
        min_support_param, rare_train_only_param,
        DeployAfterRegister, EndpointNameParam, InstanceTypeParam, InitialInstanceCount
    ],
    steps=[split_step] + branches + [deploy_condition_step],
    sagemaker_session=p_sess,
)

pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.arn
