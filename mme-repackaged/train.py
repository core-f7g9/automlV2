# === Cleaned SageMaker Pipeline: Autopilot (XGBoost-only) with MME Deployment ===

import os
import time
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep

region = boto3.Session().region_name
p_sess = PipelineSession()
role_arn = sagemaker.get_execution_role()
sm_sess = sagemaker.Session()

CLIENT_NAME = "client1"
PROJECT_NAME = f"{CLIENT_NAME}-autopilot-v1"
OUTPUT_PREFIX = "mlops"
BUCKET = sm_sess.default_bucket()
INPUT_S3CSV = f"s3://{BUCKET}/input/data.csv"

TARGET_COLS = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]
INPUT_FEATURES = ["VendorName", "LineDescription", "ClubNumber"]

bucket_param = ParameterString("Bucket", default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV", default_value=INPUT_S3CSV)
val_frac_param = ParameterFloat("ValFrac", default_value=0.20)
seed_param = ParameterInteger("RandomSeed", default_value=42)
min_support_param = ParameterInteger("MinSupport", default_value=5)
rare_train_only_param = ParameterBoolean("RareTrainOnly", default_value=True)
DeployAfterRegister = ParameterBoolean("DeployAfterRegister", default_value=True)
EndpointNameParam = ParameterString("EndpointName", default_value=f"{PROJECT_NAME}-codes-mme")
InstanceTypeParam = ParameterString("InstanceType", default_value="ml.m5.large")
InitialInstanceCount = ParameterInteger("InitialInstanceCount", default_value=1)

img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
split_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs += [
        ProcessingOutput(output_name=f"train_{tgt}", source=f"/opt/ml/processing/output/{tgt}/train"),
        ProcessingOutput(output_name=f"validation_{tgt}", source=f"/opt/ml/processing/output/{tgt}/validation")
    ]

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=split_processor,
    inputs=[ProcessingInput(source=input_s3_csv_param, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac_param.to_string(),
        "--random_seed", seed_param.to_string(),
        "--min_support", min_support_param.to_string(),
        "--rare_train_only", rare_train_only_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d")
)

branches = []
best_images = {}
best_datas = {}

for tgt in TARGET_COLS:
    train_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"train_{tgt}"].S3Output.S3Uri
    val_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{tgt}"].S3Output.S3Uri

    auto_inputs = [
        AutoMLInput(inputs=train_s3, channel_type="training", content_type="text/csv;header=present", target_attribute_name=tgt),
        AutoMLInput(inputs=val_s3, channel_type="validation", content_type="text/csv;header=present", target_attribute_name=tgt)
    ]

    automl = AutoML(
        role=role_arn,
        sagemaker_session=p_sess,
        target_attribute_name=tgt,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/autopilot-output/{tgt}/",
        problem_type="MulticlassClassification",
        job_objective={"MetricName": "Accuracy"},
        mode="ENSEMBLING",
        max_candidates=5,
        max_runtime_per_training_job_in_seconds=1800,
        total_job_runtime_in_seconds=6 * 3600,
        include_algorithms=["xgboost"]
    )

    step = AutoMLStep(name=f"RunAutopilotV1_{tgt}", step_args=automl.fit(inputs=auto_inputs))
    branches.append(step)

    best_images[tgt] = step.properties.BestCandidate.InferenceContainers[0].Image
    best_datas[tgt] = step.properties.BestCandidate.InferenceContainers[0].ModelDataUrl

deploy_lambda_name = f"{PROJECT_NAME}-deploy-mme"
MME_MODELS_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/models/"

deploy_lam = Lambda(
    function_name=deploy_lambda_name,
    execution_role_arn=role_arn,
    script="deploy_mme_from_autopilot.py",
    handler="deploy_mme_from_autopilot.handler",
    timeout=600,
    memory_size=512,
    environment={"Variables": {"EXEC_ROLE_ARN": role_arn}},
)

target_names_csv = ",".join(TARGET_COLS)
target_images_csv = Join(on=",", values=[best_images[t].to_string() for t in TARGET_COLS])
target_datas_csv = Join(on=",", values=[best_datas[t] for t in TARGET_COLS])

deploy_step = LambdaStep(
    name="DeployAllOnOneInstance_MME",
    lambda_func=deploy_lam,
    inputs={
        "EndpointName": EndpointNameParam.to_string(),
        "InstanceType": InstanceTypeParam.to_string(),
        "InitialInstanceCount": InitialInstanceCount.to_string(),
        "ModelsPrefix": MME_MODELS_PREFIX,
        "TargetNamesCSV": target_names_csv,
        "TargetImagesCSV": target_images_csv,
        "TargetModelDatasCSV": target_datas_csv,
    },
    outputs=[LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)]
)

deploy_condition_step = ConditionStep(
    name="MaybeDeployAllOnOneInstance_MME",
    conditions=[ConditionEquals(left=DeployAfterRegister, right=True)],
    if_steps=[deploy_step],
    else_steps=[]
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
print("Pipeline execution started:", execution.arn)
