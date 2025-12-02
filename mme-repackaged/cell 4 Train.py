# ============================
# Cell 4: Split Step
# ============================

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

sk_img = sagemaker.image_uris.retrieve("sklearn", region, "1.2-1")

split_processor = ScriptProcessor(
    image_uri=sk_img,
    role=role_arn,
    instance_type="ml.m5.large",
    command=["python3"],
    sagemaker_session=p_sess
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.append(
        ProcessingOutput(output_name=f"{tgt}_train", source=f"/opt/ml/processing/output/{tgt}/train")
    )
    processing_outputs.append(
        ProcessingOutput(output_name=f"{tgt}_val", source=f"/opt/ml/processing/output/{tgt}/validation")
    )

split_step = ProcessingStep(
    name="SplitPerTarget",
    processor=split_processor,
    code="prepare_per_target_splits.py",
    inputs=[ProcessingInput(source=INPUT_S3CSV, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
)

# ============================
# Cell 5 (FIXED): AutoML V2 training steps
# ============================

from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum

automl_lambda = Lambda(
    function_name=f"{PROJECT_NAME}-automl-v2",
    execution_role_arn=role_arn,
    script="lambda_automl_v2.py",
    handler="handler",
    timeout=900,
    memory_size=512
)

train_steps = []
model_data_outputs = {}
image_outputs = {}

for tgt in TARGET_COLS:

    # ⭐ FIX: Correct output names
    train_output = split_step.properties.ProcessingOutputConfig.Outputs[f"{tgt}_train"].S3Output.S3Uri
    val_output   = split_step.properties.ProcessingOutputConfig.Outputs[f"{tgt}_val"].S3Output.S3Uri

    step = LambdaStep(
        name=f"Train_{tgt}_AutoMLV2",
        lambda_func=automl_lambda,
        inputs={
            "Target": tgt,
            "TrainS3": train_output,
            "ValS3": val_output,
            "RoleArn": role_arn,
            "OutputPath": f"s3://{BUCKET}/{OUTPUT_PREFIX}/automl-v2-xgb/{tgt}/"
        },
        outputs=[
            LambdaOutput("ModelDataUrl", output_type=LambdaOutputTypeEnum.String),
            LambdaOutput("ImageUri", output_type=LambdaOutputTypeEnum.String),
        ],
    )

    train_steps.append(step)
    model_data_outputs[tgt] = step.properties.Outputs["ModelDataUrl"]
    image_outputs[tgt]      = step.properties.Outputs["ImageUri"]

# ============================
# Cell 6: Register Each Model
# ============================

from sagemaker.workflow.step_collections import RegisterModel

register_steps = []

for tgt in TARGET_COLS:
    reg = RegisterModel(
        name=f"Register_{tgt}",
        model_data=model_data_outputs[tgt],
        image_uri=image_outputs[tgt],
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name=f"{CLIENT_NAME}-{tgt}-models",
        approval_status="Approved"
    )
    register_steps.append(reg)

# ============================
# Cell 7: Build Pipeline A
# ============================

from sagemaker.workflow.pipeline import Pipeline

pipeline_a = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-A",
    steps=[split_step] + train_steps + register_steps,
    sagemaker_session=p_sess,
)

pipeline_a.upsert(role_arn=role_arn)
execution = pipeline_a.start()

execution
