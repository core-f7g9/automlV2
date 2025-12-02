# ============================================================
# Cell 4: Pipeline definition — preprocess → AutoML → repack → register
# ============================================================
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger, ParameterBoolean
)
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig, Step
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.lambda_helper import Lambda
from sagemaker.model import Model
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.inputs import TrainingInput

import boto3

pipeline_session = PipelineSession()
role = role_arn

# -------------------
# Pipeline Parameters
# -------------------
param_val_frac        = ParameterFloat("ValFrac", default_value=VAL_FRAC_DEFAULT)
param_min_support     = ParameterInteger("MinSupport", default_value=MIN_SUPPORT_DEFAULT)
param_rare_train_only = ParameterString("RareTrainOnly", default_value=str(RARE_TRAIN_ONLY_DEFAULT).lower())

param_targets_csv      = ParameterString("TargetsCSV", default_value=",".join(TARGET_COLS))
param_input_feats_csv  = ParameterString("InputFeatsCSV", default_value=",".join(INPUT_FEATURES))
param_input_csv_s3     = ParameterString("InputCSV", default_value=INPUT_S3CSV)

# -----------------------------------
# Step 1: Preprocessing (per-target)
# -----------------------------------
proc = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=f"{PROJECT_NAME}-prep",
    role=role
)

step_preprocess = ProcessingStep(
    name="PreprocessPerTarget",
    processor=proc,
    inputs=[
        ProcessingInput(
            source=param_input_csv_s3,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="splits",
            source="/opt/ml/processing/output",
            destination=f"{DATA_PREFIX}/splits"
        )
    ],
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", param_targets_csv,
        "--input_features_csv", param_input_feats_csv,
        "--val_frac", param_val_frac,
        "--random_seed", "42",
        "--min_support", param_min_support,
        "--rare_train_only", param_rare_train_only
    ],
)

# ----------------------------------------------------------
# Step 2: AutoML per target — forced XGBoost via boto3 script
# ----------------------------------------------------------
auto_ml_steps = []
targets = TARGET_COLS

for tgt in targets:

    automl_proc = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name=f"{PROJECT_NAME}-automl-{tgt}"
    )

    prefix_train = f"{DATA_PREFIX}/splits/{tgt}/train"

    step_automl = ProcessingStep(
        name=f"AutoML_XGB_{tgt}",
        processor=automl_proc,
        code="run_automl_xgb.py",
        job_arguments=[
            "--job_name", f"{PROJECT_NAME}-{tgt}-automl",
            "--train_s3_uri", prefix_train,
            "--target_col", tgt,
            "--role_arn", role_arn,
            "--output_prefix", f"{DATA_PREFIX}/automl/{tgt}"
        ],
        property_files=[
            PropertyFile(
                name=f"BestCandidate_{tgt}",
                output_name="processing_output",
                path="best_candidate.json"
            )
        ]
    )

    auto_ml_steps.append(step_automl)

# ---------------------------------------------------
# Step 3: Extract best candidate → get artifact + image
# ---------------------------------------------------
extract_steps = []
for tgt, automl_step in zip(targets, auto_ml_steps):
    
    extractor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name=f"{PROJECT_NAME}-extract-{tgt}"
    )

    step_extract = ProcessingStep(
        name=f"ExtractBest_{tgt}",
        processor=extractor,
        code="extract_best_candidate.py",
        job_arguments=[
            "--best_json", f"/opt/ml/processing/input/{automl_step.properties.ProcessingOutputConfig.Outputs['processing_output'].S3Output.S3Uri}/best_candidate.json"
        ],
        outputs=[
            ProcessingOutput(
                output_name="artifact",
                source="/opt/ml/processing/output",
                destination=f"{DATA_PREFIX}/best/{tgt}"
            )
        ],
        inputs=[]
    )

    extract_steps.append(step_extract)

# ---------------------------------------------------
# Step 4: Repack for MME
# ---------------------------------------------------
repack_steps = []
for tgt, extract_step in zip(targets, extract_steps):

    repack_proc = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name=f"{PROJECT_NAME}-repack-{tgt}"
    )

    step_repack = ProcessingStep(
        name=f"RepackMME_{tgt}",
        processor=repack_proc,
        code="repack_for_mme.py",
        job_arguments=[
            "--model_artifact",
            f"{DATA_PREFIX}/best/{tgt}/model_artifact.tar.gz",
            "--inference_script", "inference.py",
        ],
        outputs=[
            ProcessingOutput(
                output_name="mme_tar",
                source="/opt/ml/processing/output",
                destination=f"{DATA_PREFIX}/mme/{tgt}"
            )
        ]
    )
    repack_steps.append(step_repack)

# ---------------------------------------------------
# Step 5: Register Model
# ---------------------------------------------------
register_steps = []
for tgt, repack_step in zip(targets, repack_steps):

    model = Model(
        image_uri="FIX-LATER",
        model_data=repack_step.properties.ProcessingOutputConfig.Outputs["mme_tar"].S3Output.S3Uri,
        role=role,
        entry_point="inference.py",
        sagemaker_session=pipeline_session
    )

    step_reg = RegisterModel(
        name=f"Register_{tgt}",
        model=model,
        model_package_group_name=f"{PROJECT_NAME}-{tgt}",
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"]
    )

    register_steps.append(step_reg)

# ----------------------
# Build Pipeline
# ----------------------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[
        param_val_frac, param_min_support, param_rare_train_only,
        param_targets_csv, param_input_feats_csv, param_input_csv_s3
    ],
    steps=[step_preprocess] + auto_ml_steps + extract_steps + repack_steps + register_steps,
    sagemaker_session=pipeline_session
)

pipeline_json = pipeline.definition()
print(pipeline_json)

