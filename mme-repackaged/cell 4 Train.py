# ============================================================
# Cell 4 — Pipeline: Preprocess → Train (AutoML V1) → Repack → Register
# ============================================================

import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

cache_config = CacheConfig(enable_caching=False)

# ============================================================
# PREPROCESSING STEP
# ============================================================

processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region="us-east-1"),
    command=["python3"],
    role=role_arn,
    instance_type="ml.m5.xlarge",
    instance_count=1
)

step_preprocess = ProcessingStep(
    name="PerTargetPreprocess",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=INPUT_S3CSV,
            destination="/opt/ml/processing/input",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/{OUTPUT_PREFIX}/preprocess/"
        )
    ],
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", "0.2",
        "--random_seed", "42",
        "--min_support", "5",
        "--rare_train_only", "true",
    ],
    cache_config=cache_config,
)

# ============================================================
# CREATE LAMBDA FUNCTIONS
# ============================================================

lambda_automl = Lambda(
    function_name=f"{CLIENT_NAME}-lambda-automl",
    execution_role=role_arn,
    script="lambda_launch_automl.py",
    handler="handler",
    timeout=900,
)

lambda_repack = Lambda(
    function_name=f"{CLIENT_NAME}-lambda-repack",
    execution_role=role_arn,
    script="lambda_repack_mme.py",
    handler="handler",
    timeout=900,
)

# ============================================================
# AUTOML + REPACK + REGISTER (loop per target)
# ============================================================

all_steps = [step_preprocess]
repacked_artifacts = {}

for tgt in TARGET_COLS:

    # -----------------------
    # AutoML step
    # -----------------------
    automl_config = {
        "AutoMLJobName": f"{CLIENT_NAME}-{tgt}-automl",
        "InputDataConfig": [{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/{OUTPUT_PREFIX}/preprocess/{tgt}/train/"
                }
            },
            "TargetAttributeName": tgt,
            "ContentType": "text/csv",
            "CompressionType": "None"
        }],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{BUCKET}/{OUTPUT_PREFIX}/automl/{tgt}/"
        },
        "RoleArn": role_arn,
        "AutoMLJobConfig": {
            "CompletionCriteria": {"MaxRuntimePerTrainingJobInSeconds": 1200},
            "Mode": "ENSEMBLING",
            "AlgorithmConfig": {"AutoMLAlgorithms": ["XGBoost"]},
        },
        "ProblemType": "MulticlassClassification"
    }

    step_automl = LambdaStep(
        name=f"AutoML_{tgt}",
        lambda_func=lambda_automl,
        inputs={"automl_config": automl_config},
        outputs=[
            LambdaOutput(output_name="best_model_artifact", output_type=LambdaOutputTypeEnum.String),
        ],
    )

    # -----------------------
    # Repack step
    # -----------------------
    step_repack = LambdaStep(
        name=f"Repack_{tgt}",
        lambda_func=lambda_repack,
        inputs={
            "model_artifact": step_automl.outputs["best_model_artifact"],
            "target": tgt,
        },
        outputs=[
            LambdaOutput(output_name="repacked_artifact", output_type=LambdaOutputTypeEnum.String),
        ],
    )

    repacked_artifacts[tgt] = step_repack.outputs["repacked_artifact"]

    # -----------------------
    # Register Model step
    # -----------------------
    model = Model(
        model_data=step_repack.outputs["repacked_artifact"],
        role=role_arn,
        image_uri=sagemaker.image_uris.retrieve("xgboost", region),
        entry_point="inference.py",
        source_dir=".",  # inference.py is local
    )

    register_step = ModelStep(
        name=f"RegisterModel_{tgt}",
        model=model,
        model_data=step_repack.outputs["repacked_artifact"],
        depends_on=[step_repack.name],
    )

    all_steps += [step_automl, step_repack, register_step]

# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    name=f"{CLIENT_NAME}-xgb-mme",
    parameters=[],
    steps=all_steps,
    sagemaker_session=p_sess,
)

print("Pipeline created — ready for execution.")
