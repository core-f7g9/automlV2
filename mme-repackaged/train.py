import boto3
import sagemaker

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.workflow.functions import Join

from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.steps import CacheConfig

# Reuse setup variables

# Pipeline parameters
bucket_param = ParameterString("Bucket", default_value=BUCKET)
input_s3_csv_param = ParameterString("InputS3CSV", default_value=INPUT_S3CSV)
val_frac_param = ParameterFloat("ValFrac", default_value=0.20)
seed_param = ParameterInteger("RandomSeed", default_value=42)
min_support_param = ParameterInteger("MinSupport", default_value=5)
rare_train_only_param = ParameterBoolean("RareTrainOnly", default_value=True)

# Step 1: Per-target splits
sk_img = sagemaker.image_uris.retrieve("sklearn", region, "1.2-1")

split_processor = ScriptProcessor(
    image_uri=sk_img,
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
        ProcessingOutput(output_name=f"validation_{tgt}", source=f"/opt/ml/processing/output/{tgt}/validation"),
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
)

automl_steps = {}
best_model_data = {}
best_model_image = {}

for tgt in TARGET_COLS:
    train_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"train_{tgt}"].S3Output.S3Uri
    val_s3   = split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{tgt}"].S3Output.S3Uri

    inputs = [
        AutoMLInput(
            inputs=train_s3,
            channel_type="training",
            target_attribute_name=tgt,
            content_type="text/csv;header=present",
        ),
        AutoMLInput(
            inputs=val_s3,
            channel_type="validation",
            target_attribute_name=tgt,
            content_type="text/csv;header=present",
        ),
    ]

    automl = AutoML(
        role=role_arn,
        sagemaker_session=p_sess,
        target_attribute_name=tgt,
        problem_type="MulticlassClassification",
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/automl-xgb/{tgt}/",
        job_objective={"MetricName": "Accuracy"},
        mode="ENSEMBLING",
        max_candidates=5,
        include_algorithms=["xgboost"],
    )

    step = AutoMLStep(
        name=f"AutoML_XGB_{tgt}",
        step_args=automl.fit(inputs)
    )
    automl_steps[tgt] = step

    best_model_data[tgt]  = step.properties.BestCandidate.InferenceContainers[0].ModelDataUrl
    best_model_image[tgt] = step.properties.BestCandidate.InferenceContainers[0].Image

register_steps = []

for tgt in TARGET_COLS:
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=automl_steps[tgt].properties.BestCandidate.Metrics[0].Value,
            content_type="application/json"
        )
    )

    register_step = RegisterModel(
        name=f"RegisterModel_{tgt}",
        estimator=None,  # not needed for AutoML
        model_data=best_model_data[tgt],
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name=f"{CLIENT_NAME}-{tgt}-models",
        image_uri=best_model_image[tgt],
        approval_status="Approved",
        model_metrics=model_metrics
    )

    register_steps.append(register_step)

pipeline_a = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-xgb",
    parameters=[
        bucket_param,
        input_s3_csv_param,
        val_frac_param,
        seed_param,
        min_support_param,
        rare_train_only_param,
    ],
    steps=[split_step] + list(automl_steps.values()) + register_steps,
    sagemaker_session=p_sess,
)

pipeline_a.upsert(role_arn=role_arn)
execution = pipeline_a.start()
print("Started Pipeline A:", execution.arn)
