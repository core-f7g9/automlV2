# ============================
# Cell 3: Pipeline build
# ============================
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import *
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet, Join

from sagemaker.estimator import Estimator
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals

p_sess = PipelineSession()
sm_sess = sagemaker.Session()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# Parameters
input_s3_param = ParameterString("InputCSV", default_value=INPUT_S3CSV)
val_frac_param = ParameterFloat("ValFrac", default_value=0.20)
seed_param     = ParameterInteger("Seed", default_value=42)
min_supp_param = ParameterInteger("MinSupport", default_value=5)
rare_param     = ParameterBoolean("RareTrainOnly", default_value=True)

deploy_param   = ParameterBoolean("Deploy", default_value=True)
endpoint_param = ParameterString("EndpointName", default_value=f"{PROJECT_NAME}-mme")
inst_type_param = ParameterString("InstanceType", default_value="ml.m5.large")

# Preprocessing Step
sk_img = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
processor = ScriptProcessor(
    role=role,
    image_uri=sk_img,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

proc_outputs = []
propfiles = []

for t in TARGET_COLS:
    proc_outputs.append(ProcessingOutput(output_name=f"train_{t}", source=f"/opt/ml/processing/output/{t}/train"))
    proc_outputs.append(ProcessingOutput(output_name=f"val_{t}", source=f"/opt/ml/processing/output/{t}/validation"))
    propfiles.append(
        PropertyFile(name=f"{t}_Classes", output_name=f"train_{t}", path="classes.json")
    )

split_step = ProcessingStep(
    name="Preprocess",
    processor=processor,
    inputs=[ProcessingInput(source=input_s3_param, destination="/opt/ml/processing/input")],
    outputs=proc_outputs,
    property_files=propfiles,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac_param.to_string(),
        "--random_seed", seed_param.to_string(),
        "--min_support", min_supp_param.to_string(),
        "--rare_train_only", rare_param.to_string(),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output"
    ]
)

# Training Steps
xgb_img = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")

train_steps = []
model_artifacts = []

for t in TARGET_COLS:

    nclass = JsonGet(
        step_name=split_step.name,
        property_file=next(p for p in propfiles if p.name == f"{t}_Classes"),
        json_path="num_class"
    )

    est = Estimator(
        image_uri=xgb_img,
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/models/{t}/",
        sagemaker_session=p_sess
    )

    est.set_hyperparameters(
        objective="multi:softmax",
        num_round=200,
        num_class=nclass
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

# Lambda Deployment Step
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
        "EndpointName": endpoint_param,
        "InstanceType": inst_type_param,
        "InitialInstanceCount": "1",
        "TargetNamesCSV": ",".join(TARGET_COLS),
        "TargetImagesCSV": ",".join([xgb_img]*len(TARGET_COLS)),
        "TargetModelDatasCSV": Join(on=",", values=model_artifacts),
        "ModelsPrefix": f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/models/"
    },
    outputs=[LambdaOutput("status", LambdaOutputTypeEnum.String)]
)

cond = ConditionStep(
    name="MaybeDeployMME",
    conditions=[ConditionEquals(left=deploy_param, right=True)],
    if_steps=[deploy_step],
    else_steps=[]
)

pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-mme",
    parameters=[input_s3_param, val_frac_param, seed_param, min_supp_param, rare_param,
                deploy_param, endpoint_param, inst_type_param],
    steps=[split_step] + train_steps + [cond],
    sagemaker_session=p_sess
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Started pipeline:", execution.arn)
