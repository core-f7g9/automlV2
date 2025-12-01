# ============================
# Cell 3: Pipeline — XGBoost + MME
# ============================
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals

region = boto3.Session().region_name
p_sess = PipelineSession()
role = sagemaker.get_execution_role()
sm_sess = sagemaker.Session()

BUCKET = BUCKET
TARGETS = TARGET_COLS

# -----------------------------
# Parameters
# -----------------------------
input_s3 = ParameterString("InputCSV", default_value=INPUT_S3CSV)
val_frac = ParameterFloat("ValFrac", default_value=0.20)
seed     = ParameterInteger("Seed", default_value=42)

deploy_flag = ParameterBoolean("Deploy", default_value=True)
endpoint_name = ParameterString("EndpointName", default_value=f"{PROJECT_NAME}-mme")
instance_type = ParameterString("InstanceType", default_value="ml.m5.large")

# -----------------------------
# Preprocessing Step
# -----------------------------
img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")

split_proc = ScriptProcessor(
    image_uri=img,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

proc_outputs = []
propfiles = []

for t in TARGETS:
    proc_outputs.append(ProcessingOutput(output_name=f"train_{t}", source=f"/opt/ml/processing/output/{t}/train"))
    proc_outputs.append(ProcessingOutput(output_name=f"val_{t}",   source=f"/opt/ml/processing/output/{t}/validation"))
    propfiles.append(
        PropertyFile(
            name=f"{t}ClassFile",
            output_name=f"train_{t}",
            path="classes.json"
        )
    )

split_step = ProcessingStep(
    name="PreprocessData",
    processor=split_proc,
    inputs=[ProcessingInput(source=input_s3, destination="/opt/ml/processing/input")],
    outputs=proc_outputs,
    property_files=propfiles,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGETS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac,
        "--random_seed", seed,
        "--min_support", "5",
        "--rare_train_only", "true",
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output"
    ]
)

# -----------------------------
# Training Steps — XGBoost
# -----------------------------
xgb_img = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")

train_steps = []
model_artifacts = []

for t in TARGETS:

    num_class = JsonGet(
        step_name=split_step.name,
        property_file=next(p for p in propfiles if p.name == f"{t}ClassFile"),
        json_path="num_class"
    )

    est = Estimator(
        image_uri=xgb_img,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=p_sess,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/models/{t}/"
    )
    est.set_hyperparameters(
        objective="multi:softmax",
        num_round=100,
        num_class=num_class
    )

    t_step = TrainingStep(
        name=f"Train_{t}",
        estimator=est,
        inputs={
            "train": TrainingInput(
                split_step.properties.ProcessingOutputConfig.Outputs[f"train_{t}"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                split_step.properties.ProcessingOutputConfig.Outputs[f"val_{t}"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    train_steps.append(t_step)
    model_artifacts.append(t_step.properties.ModelArtifacts.S3ModelArtifacts)

# -----------------------------
# Lambda MME Deployment
# -----------------------------
deploy_lambda = Lambda(
    function_name=f"{PROJECT_NAME}-deploy-mme",
    execution_role_arn=role,
    script="deploy_mme_from_autopilot.py",
    handler="deploy_mme_from_autopilot.handler",
    timeout=600,
    memory_size=512
)

deploy_step = LambdaStep(
    name="DeployMME",
    lambda_func=deploy_lambda,
    inputs={
        "EndpointName": endpoint_name,
        "InstanceType": instance_type,
        "InitialInstanceCount": "1",
        "TargetNamesCSV": ",".join(TARGETS),
        "TargetImagesCSV": ",".join([xgb_img]*len(TARGETS)),
        "TargetModelDatasCSV": Join(on=",", values=model_artifacts),
        "ModelsPrefix": f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/models/"
    },
    outputs=[LambdaOutput("status", LambdaOutputTypeEnum.String)]
)

cond = ConditionStep(
    name="MaybeDeployMME",
    conditions=[ConditionEquals(left=deploy_flag, right=True)],
    if_steps=[deploy_step],
    else_steps=[]
)

# -----------------------------
# Build pipeline
# -----------------------------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-xgb-mme",
    parameters=[input_s3, val_frac, seed, deploy_flag, endpoint_name, instance_type],
    steps=[split_step] + train_steps + [cond],
    sagemaker_session=p_sess
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Pipeline started:", execution.arn)
