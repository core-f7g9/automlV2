# Full SageMaker Pipeline: Setup + Preprocess + Train for XGBoost MME (Jupyter Notebook Friendly)

# =========================
# Cell 1: Setup variables
# =========================
import os
import boto3
import sagemaker
from urllib.parse import urlparse
from botocore.exceptions import ClientError

region = boto3.Session().region_name
sm_sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# ---- Client/project knobs
CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-xgboost-mme"
OUTPUT_PREFIX = "mlops"

# ---- Data locations
BUCKET       = sm_sess.default_bucket()
INPUT_S3CSV  = f"s3://{BUCKET}/input/data.csv"
DATA_PREFIX  = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

# ---- Targets and features
TARGET_COLS    = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]
INPUT_FEATURES = ["VendorName", "LineDescription", "ClubNumber"]

# ---- Utility: Check input
s3 = boto3.client("s3", region_name=region)
def _parse_s3(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")

csv_bucket, csv_key = _parse_s3(INPUT_S3CSV)
try:
    s3.head_object(Bucket=csv_bucket, Key=csv_key)
except ClientError as e:
    code = e.response.get("Error", {}).get("Code", "")
    if code in ("404", "NoSuchKey", "NotFound"):
        raise FileNotFoundError(f"CSV not found at {INPUT_S3CSV}.") from e
    raise RuntimeError(f"Could not access {INPUT_S3CSV}: {e}") from e

print("Region:", region)
print("Role:", role)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Targets:", TARGET_COLS)
print("Input features:", INPUT_FEATURES)
print("Project:", PROJECT_NAME)

# ============================
# Cell 2: Write preprocessing script
# ============================
import textwrap
split_script = textwrap.dedent("""
import argparse
import os
import pandas as pd
import numpy as np

def per_target_split(df, target_col, input_feats, val_frac=0.2, seed=42):
    df_t = df.dropna(subset=[target_col]).copy()
    df_t[target_col] = df_t[target_col].astype(str)
    rng = np.random.RandomState(seed)
    val_idx = rng.rand(len(df_t)) < val_frac
    train = df_t[~val_idx]
    val = df_t[val_idx]
    keep_cols = [target_col] + input_feats
    return train[keep_cols], val[keep_cols]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_csv", type=str, required=True)
    parser.add_argument("--input_features_csv", type=str, required=True)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--mounted_input_dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--headerless", type=str, default="false")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.mounted_input_dir, "data.csv"), low_memory=False)
    targets = [t.strip() for t in args.targets_csv.split(",") if t.strip()]
    input_feats = [f.strip() for f in args.input_features_csv.split(",") if f.strip()]
    headerless = args.headerless.lower() in ("true", "1", "yes")

    for tgt in targets:
        train, val = per_target_split(df, tgt, input_feats, val_frac=args.val_frac)
        out_train = os.path.join(args.output_dir, tgt, "train")
        out_val = os.path.join(args.output_dir, tgt, "validation")
        os.makedirs(out_train, exist_ok=True)
        os.makedirs(out_val, exist_ok=True)
        train.to_csv(os.path.join(out_train, "train.csv"), index=False, header=not headerless)
        val.to_csv(os.path.join(out_val, "validation.csv"), index=False, header=not headerless)

if __name__ == "__main__":
    main()
""")
with open("prepare_per_target_splits.py", "w") as f:
    f.write(split_script)
print("Wrote prepare_per_target_splits.py")

# ============================
# Cell 3: Pipeline definition — Preprocess + Train + Deploy
# ============================
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterBoolean
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.functions import Join

p_sess = PipelineSession()

bucket_param = ParameterString("Bucket", default_value=BUCKET)
input_csv_param = ParameterString("InputS3CSV", default_value=INPUT_S3CSV)
DeployAfterRegister = ParameterBoolean("DeployAfterRegister", default_value=True)
EndpointNameParam = ParameterString("EndpointName", default_value=f"{PROJECT_NAME}-mme")
InstanceTypeParam = ParameterString("InstanceType", default_value="ml.m5.large")
InitialInstanceCount = ParameterInteger("InitialInstanceCount", default_value=1)

XGB_IMAGE = sagemaker.image_uris.retrieve("xgboost", region, version="1.3-1")

# Split step
split_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

processing_outputs = []
for tgt in TARGET_COLS:
    processing_outputs.append(ProcessingOutput(output_name=f"train_{tgt}", source=f"/opt/ml/processing/output/{tgt}/train"))
    processing_outputs.append(ProcessingOutput(output_name=f"validation_{tgt}", source=f"/opt/ml/processing/output/{tgt}/validation"))

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=split_processor,
    inputs=[ProcessingInput(source=input_csv_param, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", "0.2",
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
        "--headerless", "true"
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d")
)

# Training steps
branches = []
model_s3_uris = {}
for tgt in TARGET_COLS:
    train_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"train_{tgt}"].S3Output.S3Uri
    val_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{tgt}"].S3Output.S3Uri

    xgb = XGBoost(
        entry_point=None,
        framework_version="1.3-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        image_uri=XGB_IMAGE,
        sagemaker_session=p_sess,
        hyperparameters={"objective": "multi:softprob", "num_class": 10, "num_round": 50},
    )

    train_step = TrainingStep(
        name=f"TrainModel_{tgt}",
        estimator=xgb,
        inputs={
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(val_s3, content_type="text/csv")
        }
    )

    model_s3_uris[tgt] = train_step.properties.ModelArtifacts.S3ModelArtifacts
    branches.append(train_step)

# Deployment step
MME_MODELS_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/models/"

deploy_lambda_name = f"{PROJECT_NAME}-deploy-mme"
deploy_lam = Lambda(
    function_name=deploy_lambda_name,
    execution_role_arn=role,
    script="deploy_xgb_mme.py",
    handler="deploy_xgb_mme.handler",
    timeout=600,
    memory_size=512,
    environment={"Variables": {"EXEC_ROLE_ARN": role}},
)

target_names_csv = ",".join(TARGET_COLS)
target_datas_csv = Join(on=",", values=[model_s3_uris[t] for t in TARGET_COLS])

deploy_step = LambdaStep(
    name="DeployMME",
    lambda_func=deploy_lam,
    inputs={
        "EndpointName": EndpointNameParam,
        "InstanceType": InstanceTypeParam,
        "InitialInstanceCount": InitialInstanceCount,
        "ModelsPrefix": MME_MODELS_PREFIX,
        "TargetNamesCSV": target_names_csv,
        "TargetModelDatasCSV": target_datas_csv,
        "XGBoostImage": XGB_IMAGE
    },
    outputs=[LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)]
)

pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[bucket_param, input_csv_param, DeployAfterRegister, EndpointNameParam, InstanceTypeParam, InitialInstanceCount],
    steps=[split_step] + branches + [deploy_step],
    sagemaker_session=p_sess
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Pipeline execution ARN:", execution.arn)
