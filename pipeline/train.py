# ============================================
# Cell 3: Build & run the SageMaker Pipeline
# ============================================
import json
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel

p_sess = PipelineSession()

# --------- pipeline parameters (reusable per client) ----------
bucket_param       = ParameterString("Bucket",       default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV",   default_value=INPUT_S3CSV)
target_col_param   = ParameterString("TargetCol",    default_value=TARGET_COL)
val_frac_param     = ParameterFloat( "ValFrac",      default_value=0.2)
seed_param         = ParameterInteger("RandomSeed",  default_value=42)
problem_type_param = ParameterString("ProblemType",  default_value="BinaryClassification")
objective_param    = ParameterString("Objective",    default_value="F1")
role_param         = ParameterString("ExecutionRoleArn", default_value=role_arn)

# --------- Step 1: Processing (split) ----------
img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
script_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

split_step = ProcessingStep(
    name="SplitTrainValidation",
    processor=script_processor,
    inputs=[ProcessingInput(source=input_s3_csv_param, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(output_name="out", source="/opt/ml/processing/output")],
    code="sql_to_s3_and_split.py",
    job_arguments=[
        "--input_s3_csv", input_s3_csv_param,
        "--target_col",   target_col_param,
        "--val_frac",     val_frac_param.to_string(),
        "--random_seed",  seed_param.to_string(),
        "--output_dir",   "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
)

# Build S3 paths to train/validation at execution time (no Python string concat!)
train_s3_uri = Join(on="", values=[
    split_step.properties.ProcessingOutputConfig.Outputs["out"].S3Output.S3Uri,
    "/train.csv",
])
# (validation_s3_uri available if you need it)
# validation_s3_uri = Join(on="", values=[
#     split_step.properties.ProcessingOutputConfig.Outputs["out"].S3Output.S3Uri,
#     "/validation.csv",
# ])

# --------- Step 2: LambdaStep to run Autopilot V2 ----------
lambda_src = r"""
import json, time, boto3
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
    # wait
    while True:
        d = sm.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        st = d["AutoMLJobStatus"]
        if st in ("Completed","Failed","Stopped"):
            break
        time.sleep(30)
    if st != "Completed":
        raise RuntimeError(f"Autopilot V2 job {job_name} ended with {st}")

    best = d["AutoMLJobBestCandidate"]
    best_image = best["InferenceContainerDefinitions"][0]["Image"]
    best_art   = best["CandidateProperties"]["CandidateArtifactLocations"]["ModelArtifacts"]

    # return top-level fields that match LambdaStep outputs
    return {
        "BestImageUri": best_image,
        "BestModelArtifacts": best_art
    }
"""

# keep names short to avoid OS/name length issues
SAFE_NAME     = "c1-apv2"
lambda_name   = f"{SAFE_NAME}-runner"
auto_job_name = f"{SAFE_NAME}-{int(time.time())}"

lam = Lambda(
    function_name=lambda_name,
    execution_role_arn=role_arn,
    script=lambda_src,
    handler="index.handler",
    timeout=900,
    memory_size=512,
)

auto_step = LambdaStep(
    name="RunAutopilotV2",
    lambda_func=lam,
    inputs={
        "job_name":    auto_job_name,
        "role_arn":    role_param,               # ParameterString ok
        "problem_type": problem_type_param,
        "objective":    objective_param,
        "train_s3":     train_s3_uri,            # property from previous step
        "target_col":   target_col_param,
        "output_s3":    f"s3://{BUCKET}/{OUTPUT_PREFIX}/autopilot-output/",  # pure string is fine
    },
    outputs=[
        LambdaOutput(output_name="BestImageUri",       output_type=LambdaOutputTypeEnum.String),
        LambdaOutput(output_name="BestModelArtifacts", output_type=LambdaOutputTypeEnum.String),
    ],
)

# read Lambda outputs via properties API
best_image = auto_step.properties.Outputs["BestImageUri"]
best_data  = auto_step.properties.Outputs["BestModelArtifacts"]

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
    description="Best model from Autopilot V2",
)

# --------- Build & start the pipeline ----------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[bucket_param, input_s3_csv_param, target_col_param,
                val_frac_param, seed_param, problem_type_param, objective_param, role_param],
    steps=[split_step, auto_step, register_step],
    sagemaker_session=p_sess,
)

pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.arn