import boto3, sagemaker, json, time
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

p_sess = sagemaker.workflow.pipeline_context.PipelineSession()
project_name = "client1-autopilot-v2"
output_prefix = "mlops"
pipeline_name = f"{project_name}-pipeline"

# ---- Pipeline parameters (so you can reuse per client) ----
bucket       = ParameterString("Bucket", default_value=BUCKET)
input_s3_csv = ParameterString("InputS3CSV", default_value=INPUT_S3CSV)
target_col   = ParameterString("TargetCol", default_value=TARGET_COL)
val_frac     = ParameterFloat("ValFrac", default_value=0.2)
random_seed  = ParameterInteger("RandomSeed", default_value=42)
problem_type = ParameterString("ProblemType", default_value="BinaryClassification")
objective    = ParameterString("Objective", default_value="F1")
role_param   = ParameterString("ExecutionRoleArn", default_value=role_arn)

# ---- Step 1: processing (split) ----
script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

split_step = ProcessingStep(
    name="SplitTrainValidation",
    processor=script_processor,
    inputs=[ProcessingInput(source=input_s3_csv, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(output_name="out", source="/opt/ml/processing/output")],
    code="sql_to_s3_and_split.py",
    job_arguments=[
        "--input_s3_csv", input_s3_csv,
        "--target_col", target_col,
        "--val_frac", str(val_frac),
        "--random_seed", str(random_seed),
        "--output_dir", "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

train_s3_uri = split_step.properties.ProcessingOutputConfig.Outputs["out"].S3Output.S3Uri + "/train.csv"

# ---- Step 2: Lambda to run Autopilot V2 ----
lambda_src = r"""
import os, json, time, boto3
sm = boto3.client("sagemaker")
def handler(event, context):
    job_name = event["job_name"]
    req = {
        "AutoMLJobName": job_name,
        "AutoMLJobInputDataConfig": [{
            "ChannelType": "training",
            "CompressionType": "None",
            "ContentType": "text/csv",
            "S3Input": {"S3Uri": event["train_s3"]}
        }],
        "OutputDataConfig": {"S3OutputPath": event["output_s3"]},
        "RoleArn": event["role_arn"],
        "AutoMLProblemType": event["problem_type"],
        "AutoMLJobObjective": {"MetricName": event["objective"]},
        "AutoMLProblemTypeConfig": {
            "TabularJobConfig": {
                "TargetAttributeName": event["target_col"],
                "CompletionCriteria": {"MaxCandidates": 10},
                "Mode": "ENSEMBLING"
            }
        }
    }
    sm.create_auto_ml_job_v2(**req)
    while True:
        d = sm.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        st = d["AutoMLJobStatus"]
        if st in ("Completed","Failed","Stopped"):
            break
        time.sleep(30)
    if st != "Completed":
        raise RuntimeError(f"Autopilot V2 job {job_name} ended with {st}")
    best = d["AutoMLJobBestCandidate"]
    return {"statusCode": 200, "body": json.dumps({
        "inference_image_uri": best["InferenceContainerDefinitions"][0]["Image"],
        "model_artifacts_s3": best["CandidateProperties"]["CandidateArtifactLocations"]["ModelArtifacts"]
    })}"""

lam = Lambda(
    function_name=f"{project_name}-autopilotv2",
    execution_role_arn=role_arn,
    script=lambda_src,
    handler="index.handler",
    timeout=900,
    memory_size=512,
)

auto_job_name = f"{project_name}-apv2-{int(time.time())}"

auto_step = LambdaStep(
    name="RunAutopilotV2",
    lambda_func=lam,
    inputs={
        "job_name": auto_job_name,
        "role_arn": role_arn,
        "problem_type": problem_type,
        "objective": objective,
        "train_s3": train_s3_uri,
        "target_col": target_col,
        "output_s3": f"s3://{BUCKET}/{output_prefix}/autopilot-output/",
    },
    outputs=[LambdaOutput(output_name="payload", output_type="String")],
)

best_image = JsonGet(step_name=auto_step.name, property_file=None, json_path="$.inference_image_uri")
best_data  = JsonGet(step_name=auto_step.name, property_file=None, json_path="$.model_artifacts_s3")

# ---- Step 3: Register model ----
model_to_register = Model(image_uri=best_image, model_data=best_data, role=role_arn, sagemaker_session=p_sess)

register_step = RegisterModel(
    name="RegisterBestModel",
    model=model_to_register,
    content_types=["text/csv","application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large","ml.c5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=f"{project_name}-pkg-group",
    approval_status="Approved",
    description="Best model from Autopilot V2",
)

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[bucket, input_s3_csv, target_col, val_frac, random_seed, problem_type, objective, role_param],
    steps=[split_step, auto_step, register_step],
    sagemaker_session=p_sess,
)

# Upsert + start
pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.arn
