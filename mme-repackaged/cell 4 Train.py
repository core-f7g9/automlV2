# ----------------------------------------------------------------------------
# STEP 4: Train XGBoost using AutoML hyperparameters
# ----------------------------------------------------------------------------
clean_hp_uri = f"{DATA_PREFIX}/hp_clean/{tgt}/clean_hp.json"

# Load hyperparameters dynamically via pipeline property file
from sagemaker.workflow.properties import PropertyFile

hp_prop_file = PropertyFile(
    name=f"CleanHPFile_{tgt}",
    output_name="clean_hp",
    path="clean_hp.json"
)

# The Estimator will receive these hyperparameters at runtime
xgb_estimator = Estimator(
    image_uri=XGB_IMAGE,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"{DATA_PREFIX}/xgb/{tgt}",
    sagemaker_session=pipeline_session,
)

# Set hyperparameters automatically using pipeline property file
xgb_estimator.set_hyperparameters(
    **{
        # Pipeline property reference:
        # Loaded as JSON dict from clean_hp.json
        "hp": step_load_hp.properties.ProcessingOutputConfig.Outputs["clean_hp"].S3Output.S3Uri
    }
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
    },
    property_files=[hp_prop_file]
)

steps.append(step_train)
