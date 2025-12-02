# ============================
# Cell 4 — Pipeline A (Corrected)
# ============================

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.steps import CacheConfig

# -------------------------------
# Parameters
# -------------------------------
val_frac_param        = ParameterFloat("ValFrac", default_value=0.20)
seed_param            = ParameterInteger("Seed", default_value=42)
min_support_param     = ParameterInteger("MinSupport", default_value=5)
rare_train_only_param = ParameterBoolean("RareTrainOnly", default_value=True)

# -------------------------------
# Preprocessing (Hybrid TF-IDF)
# -------------------------------
sk_img = sagemaker.image_uris.retrieve("sklearn", region, "1.2-1")

split_proc = ScriptProcessor(
    image_uri=sk_img,
    role=role_arn,
    instance_type="ml.m5.large",
    command=["python3"],
    sagemaker_session=p_sess
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.append(
        ProcessingOutput(
            output_name=f"{tgt}_out",
            source=f"/opt/ml/processing/output/{tgt}"
        )
    )

split_step = ProcessingStep(
    name="HybridTFIDF_Preprocess",
    processor=split_proc,
    code="preprocess_hybrid.py",
    inputs=[
        ProcessingInput(
            source=INPUT_S3CSV,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=processing_outputs,
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--inputs_csv", ",".join(INPUT_FEATURES),
        "--val_frac", val_frac_param.to_string(),
        "--seed", seed_param.to_string(),
        "--min_support", min_support_param.to_string(),
        "--rare_train_only", rare_train_only_param.to_string(),
        "--input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output"
    ]
)

# -------------------------------
# Training steps per target
# -------------------------------
train_steps = []
register_steps = []

for tgt in TARGET_COLS:

    data_channel = split_step.properties.ProcessingOutputConfig.Outputs[f"{tgt}_out"].S3Output.S3Uri

    xgb_estimator = XGBoost(
        entry_point="train_xgb.py",
        source_dir="xgb_src",
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.7-1",
        py_version="py3",
        sagemaker_session=p_sess,
        hyperparameters={"target": tgt}
    )

    train_step = TrainingStep(
        name=f"Train_{tgt}",
        estimator=xgb_estimator,
        inputs={"data": data_channel},
        cache_config=CacheConfig(enable_caching=False)
    )
    train_steps.append(train_step)

    register_step = RegisterModel(
        name=f"Register_{tgt}",
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=xgb_estimator.image_uri,
        model_package_group_name=f"{CLIENT_NAME}-{tgt}-models",
        approval_status="Approved"
    )
    register_steps.append(register_step)

# -------------------------------
# Build Pipeline A
# -------------------------------
pipeline_a = Pipeline(
    name=f"{PROJECT_NAME}-pipeline-A",
    steps=[split_step] + train_steps + register_steps,
    sagemaker_session=p_sess
)

pipeline_a.upsert(role_arn=role_arn)
print("PipelineA created.")
