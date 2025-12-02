# ============================================================
# Cell 4: Pipeline definition (Preprocess → AutoML → Extract → Train XGB → Repack → Register)
# ============================================================

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger
)
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.estimator import Estimator

import boto3

pipeline_session = PipelineSession()
role = role_arn
region = boto3.Session().region_name

# --------------------
# Pipeline Parameters
# --------------------
param_val_frac        = ParameterFloat("ValFrac", default_value=VAL_FRAC_DEFAULT)
param_min_support     = ParameterInteger("MinSupport", default_value=MIN_SUPPORT_DEFAULT)
param_targets_csv     = ParameterString("TargetsCSV", default_value=",".join(TARGET_COLS))
param_input_feats_csv = ParameterString("InputFeatsCSV", default_value=",".join(INPUT_FEATURES))
param_input_csv       = ParameterString("InputCSV", default_value=INPUT_S3CSV)


# --------------------
# STEP 1: Preprocess
# --------------------
prep_proc = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=role,
    base_job_name=f"{PROJECT_NAME}-prep",
)

step_preprocess = ProcessingStep(
    name="PreprocessPerTarget",
    processor=prep_proc,
    code="prepare_per_target_splits.py",
    inputs=[
        ProcessingInput(
            source=param_input_csv,
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
    job_arguments=[
        "--targets_csv", param_targets_csv,
        "--input_features_csv", param_input_feats_csv,
        "--val_frac", param_val_frac,
        "--random_seed", "42",
        "--min_support", param_min_support,
        "--rare_train_only", "true",
    ],
)

# --------------------------------------
# STEP 2–6 (loop per target)
# --------------------------------------

steps = [step_preprocess]

XGB_IMAGE = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

for tgt in TARGET_COLS:

    # ----------------------------------------------------------------------------
    # STEP 2: AutoML hyperparameter search → produces automl_best.json
    # ----------------------------------------------------------------------------
    automl_proc = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name=f"{PROJECT_NAME}-automl-{tgt}"
    )

    step_automl = ProcessingStep(
        name=f"AutoMLSearch_{tgt}",
        processor=automl_proc,
        code="run_automl_xgb.py",
        job_arguments=[
            "--job_name", f"{PROJECT_NAME}-{tgt}-automl",
            "--train_s3_uri", f"{DATA_PREFIX}/splits/{tgt}/train",
            "--target_col", tgt,
            "--role", role,
            "--output_prefix", f"{DATA_PREFIX}/automl/{tgt}",
        ],
        outputs=[
            ProcessingOutput(
                output_name="automl_best",
                source="/opt/ml/processing/output",
                destination=f"{DATA_PREFIX}/automl_best/{tgt}"
            )
        ]
    )
    steps.append(step_automl)

    # ----------------------------------------------------------------------------
    # STEP 3: Extract hyperparameters + feature engineering config
    # ----------------------------------------------------------------------------
    extract_proc = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name=f"{PROJECT_NAME}-extract-{tgt}"
    )

    step_extract = ProcessingStep(
        name=f"ExtractAutoMLConfig_{tgt}",
        processor=extract_proc,
        code="extract_automl_config.py",
        job_arguments=[
            "--best_json", f"{DATA_PREFIX}/automl_best/{tgt}/automl_best.json",
        ],
        outputs=[
            ProcessingOutput(
                output_name="hp",
                source="/opt/ml/processing/output",
                destination=f"{DATA_PREFIX}/hp/{tgt}"
            )
        ]
    )
    steps.append(step_extract)

    # ----------------------------------------------------------------------------
    # STEP 4: Train XGBoost using AutoML hyperparameters
    # ----------------------------------------------------------------------------
    hp_uri  = f"{DATA_PREFIX}/hp/{tgt}/hyperparams.json"

    xgb_estimator = Estimator(
        image_uri=XGB_IMAGE,
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=f"{DATA_PREFIX}/xgb/{tgt}",
        sagemaker_session=pipeline_session
    )

    step_train = TrainingStep(
        name=f"TrainXGB_{tgt}",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=f"{DATA_PREFIX}/splits/{tgt}/train/train.csv",
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=f"{DATA_PREFIX}/splits/{tgt}/validation/validation.csv",
                content_type="text/csv"
            )
        }
    )
    steps.append(step_train)

    # ----------------------------------------------------------------------------
    # STEP 5: Repack for MME (model.tar.gz + inference.py → model_mme.tar.gz)
    # ----------------------------------------------------------------------------
    repack_proc = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name=f"{PROJECT_NAME}-repack-{tgt}"
    )

    trained_model_tar = step_train.properties.ModelArtifacts.S3ModelArtifacts

    step_repack = ProcessingStep(
        name=f"RepackMME_{tgt}",
        processor=repack_proc,
        code="repack_for_mme.py",
        job_arguments=[
            "--trained_model_s3", trained_model_tar,
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
    steps.append(step_repack)

    # ----------------------------------------------------------------------------
    # STEP 6: Register Model
    # ----------------------------------------------------------------------------
    model_package = RegisterModel(
        name=f"RegisterXGB_{tgt}",
        model_data=step_repack.properties.ProcessingOutputConfig.Outputs["mme_tar"].S3Output.S3Uri,
        image_uri=XGB_IMAGE,
        model_package_group_name=f"{PROJECT_NAME}-{tgt}",
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
    )
    steps.append(model_package)

# --------------------
# Build Pipeline
# --------------------
pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[
        param_val_frac, param_min_support,
        param_targets_csv, param_input_feats_csv,
        param_input_csv,
    ],
    steps=steps,
    sagemaker_session=pipeline_session
)

print("Pipeline JSON:")
print(pipeline.definition())
