from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import ModelMetrics, MetricsSource

# Pipeline params
val_frac_param    = ParameterFloat("ValFrac", default_value=0.20)
seed_param        = ParameterInteger("RandomSeed", default_value=42)
min_support_param = ParameterInteger("MinSupport", default_value=5)
rare_only_param   = ParameterBoolean("RareTrainOnly", default_value=True)

# -----------------------------
# Step 1 — Preprocessing splits
# -----------------------------
sk_img = sagemaker.image_uris.retrieve("sklearn", region, "1.2-1")

split_proc = ScriptProcessor(
    image_uri=sk_img,
    role=role_arn,
    instance_type="ml.m5.large",
    command=["python3"],
    sagemaker_session=p_sess,
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.append(
        ProcessingOutput(
            output_name=f"{tgt}_train",
            source=f"/opt/ml/processing/output/{tgt}/train"
        )
    )
    processing_outputs.append(
        ProcessingOutput(
            output_name=f"{tgt}_val",
            source=f"/opt/ml/processing/output/{tgt}/validation"
        )
    )

split_step = ProcessingStep(
    name="SplitPerTarget",
    processor=split_proc,
    code="prepare_per_target_splits.py",
    inputs=[
        ProcessingInput(source=INPUT_S3CSV, destination="/opt/ml/processing/input")
    ],
    outputs=processing_outputs,
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac_param.to_string(),
        "--random_seed", seed_param.to_string(),
        "--min_support", min_support_param.to_string(),
        "--rare_train_only", rare_only_param.to_string(),
    ]
)

# -----------------------------
# Step 2 — AutoML V2 per target
# -----------------------------
automl_lambda = Lambda(
    function_name=f"{PROJECT_NAME}-automl-v2-lambda",
    execution_role_arn=role_arn,
    script="lambda_automl_v2.py",
    handler="handler",
    timeout=900,
    memory_size=512,
)

model_data_map = {}
image_uri_map  = {}
register_steps = []

for tgt in TARGET_COLS:

    step = LambdaStep(
        name=f"TrainAutoMLV2_{tgt}",
        lambda_func=automl_lambda,
        inputs={
            "Target": tgt,
            "RoleArn": role_arn,
            "TrainS3": split_step.properties.ProcessingOutputConfig.Outputs[f"{tgt}_train"].S3Output.S3Uri,
            "ValS3":   split_step.properties.ProcessingOutputConfig.Outputs[f"{tgt}_val"].S3Output.S3Uri,
            "OutputPath": f"s3://{BUCKET}/{OUTPUT_PREFIX}/automl-v2-xgb/{tgt}/",
        },
    )

    model_data_map[tgt] = step.properties.Outputs["ModelDataUrl"]
    image_uri_map[tgt]  = step.properties.Outputs["ImageUri"]

    # Register
    reg = RegisterModel(
        name=f"Register_{tgt}",
        model_data=model_data_map[tgt],
        content_types=["text/csv"],
        response_types=["text/csv"],
        image_uri=image_uri_map[tgt],
        model_package_group_name=f"{CLIENT_NAME}-{tgt}-models",
        approval_status="Approved"
    )
    register_steps.append(reg)

pipeline_a = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-A",
    parameters=[val_frac_param, seed_param, min_support_param, rare_only_param],
    steps=[split_step] + list(model_data_map.values()) + register_steps,
    sagemaker_session=p_sess,
)

pipeline_a.upsert(role_arn=role_arn)
execution = pipeline_a.start()
print("Started Pipeline A:", execution.arn)
