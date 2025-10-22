# ============================================
# Cell 3: Build & run the SageMaker Pipeline (Autopilot V1, robust to ParameterString)
# ============================================
import json, boto3, sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel

# Autopilot V1
from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.automl_step import AutoMLStep

p_sess = PipelineSession()

# --------- pipeline parameters (keep only those that Autopilot tolerates) ----------
bucket_param       = ParameterString("Bucket",       default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV",   default_value=INPUT_S3CSV)
target_col_param   = ParameterString("TargetCol",    default_value=TARGET_COL)
val_frac_param     = ParameterFloat( "ValFrac",      default_value=0.2)
seed_param         = ParameterInteger("RandomSeed",  default_value=42)

# We’ll set problem_type & objective as literals to avoid SDK string checks on ParameterString
PROBLEM_TYPE = "MulticlassClassification"   # you asked for multiclass
OBJECTIVE    = "Accuracy"                   # or "F1" if you prefer

# --------- Step 1: Processing (split) ----------
img = sagemaker.image_uris.retrieve("sklearn", boto3.Session().region_name, version="1.2-1")
script_processor = ScriptProcessor(
    image_uri=img,
    role=role_arn,                 # plain string (from Cell 1)
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

# Build S3 path to the training CSV at execution time
train_s3_uri = Join(
    on="",
    values=[
        split_step.properties.ProcessingOutputConfig.Outputs["out"].S3Output.S3Uri,
        "/train.csv",
    ],
)

# --------- Step 2: Autopilot V1 (native AutoMLStep via step_args) ----------
# Use AutoMLInput (NOT TrainingInput), and keep values simple
auto_input = AutoMLInput(
    inputs=train_s3_uri,              # Pipeline property is OK here
    content_type="text/csv",
    channel_type="training",
    # compression defaults to None
)

# Use plain strings for role and output_path to avoid the 'ParameterString is not iterable' pitfalls
automl = AutoML(
    role=role_arn,                                            # plain string
    target_attribute_name=target_col_param,                   # ParameterString works here
    output_path=f"s3://{BUCKET}/mlops/autopilot-output/",     # plain string
    problem_type=PROBLEM_TYPE,                                # plain string
    job_objective={"MetricName": OBJECTIVE},                  # plain string
    max_candidates=10,
    mode="ENSEMBLING",
)

# IMPORTANT: pass AutoMLInput or list[AutoMLInput]
step_args = automl.fit(inputs=[auto_input])

automl_step = AutoMLStep(
    name="RunAutopilotV1",
    step_args=step_args
)

# Best candidate artifacts from the AutoML step properties (V1)
best_image = automl_step.properties.BestCandidate.InferenceContainers[0].Image
best_data  = automl_step.properties.BestCandidate.ModelArtifacts.S3ModelArtifacts

# --------- Step 3: Register best model to Model Registry ----------
model_to_register = Model(
    image_uri=best_image,
    model_data=best_data,
    role=role_arn,                    # plain string
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

# --------- Build & start the pipeline ----------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[
        bucket_param, input_s3_csv_param, target_col_param,
        val_frac_param, seed_param
    ],
    steps=[split_step, automl_step, register_step],
    sagemaker_session=p_sess,
)

pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.arn
