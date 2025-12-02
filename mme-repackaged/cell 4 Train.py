# ----------------------------------------------------------------------------
# STEP 4: Train XGBoost using AutoML hyperparameters
# ----------------------------------------------------------------------------
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

region = boto3.Session().region_name
pipeline_session = pipeline_session if "pipeline_session" in globals() else PipelineSession()
steps = steps if "steps" in globals() else []

for tgt in TARGET_COLS:
    clean_hp_uri = f"{DATA_PREFIX}/hp_clean/{tgt}/clean_hp.json"

    xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.3-1")
    xgb_estimator = XGBoost(
        image_uri=xgb_image,
        entry_point="train_xgb.py",
        source_dir=".",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=f"{DATA_PREFIX}/xgb/{tgt}",
        sagemaker_session=pipeline_session,
        framework_version="1.3-1",
        py_version="py3",
    )

    hp_source = clean_hp_uri
    if "step_load_hp" in globals():
        hp_source = step_load_hp.properties.ProcessingOutputConfig.Outputs["clean_hp"].S3Output.S3Uri

    xgb_estimator.set_hyperparameters(hp=hp_source)

    step_train = TrainingStep(
        name=f"TrainXGB_{tgt}",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=f"{DATA_PREFIX}/splits/{tgt}/train.csv",
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=f"{DATA_PREFIX}/splits/{tgt}/val.csv",
                content_type="text/csv"
            )
        }
    )

    steps.append(step_train)
