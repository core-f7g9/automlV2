# =========================
# Cell 1: Setup variables
# =========================
import os
import boto3
import sagemaker
from urllib.parse import urlparse
from botocore.exceptions import ClientError

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
role     = sagemaker.get_execution_role()

# ---- Client/project knobs
CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-xgb-mme"
OUTPUT_PREFIX = "mlops"

# ---- Data locations
BUCKET       = sm_sess.default_bucket()
INPUT_S3CSV  = f"s3://{BUCKET}/input/data.csv"
DATA_PREFIX  = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

# ---- Targets and inputs
TARGET_COLS    = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]
INPUT_FEATURES = ["VendorName", "LineDescription", "ClubNumber"]

# ---- Validation split config
VAL_FRAC_DEFAULT = 0.20

# Check input
s3 = boto3.client("s3", region_name=region)
def _parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")

csv_bucket, csv_key = _parse_s3(INPUT_S3CSV)
try:
    s3.head_object(Bucket=csv_bucket, Key=csv_key)
except ClientError as e:
    raise FileNotFoundError(f"CSV not found at {INPUT_S3CSV}") from e

print("Region:", region)
print("Role:", role)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Targets:", TARGET_COLS)
print("Input features:", INPUT_FEATURES)
print("Project:", PROJECT_NAME, "| Output prefix:", OUTPUT_PREFIX)


# =========================
# Cell 2: Write preprocess script
# =========================
preprocess_script = """
import argparse, os, glob, json
import pandas as pd
import numpy as np

def find_input_csv(mounted_dir):
    candidates = glob.glob(os.path.join(mounted_dir, "*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No CSV found in input")

def per_target_split(df, target_col, input_feats, val_frac=0.2):
    df_t = df[~df[target_col].isna()].copy()
    df_t[target_col] = df_t[target_col].astype(str)

    counts = df_t[target_col].value_counts()
    classes = counts.index.tolist()

    val_idx = []
    for cls in classes:
        g = df_t[df_t[target_col] == cls]
        n_val = max(1, int(len(g) * val_frac))
        val_idx.extend(g.sample(n=n_val, random_state=42).index)

    val = df_t.loc[val_idx]
    train = df_t.drop(index=val_idx)

    keep = input_feats + [target_col]
    return train[keep], val[keep], len(classes)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets_csv", required=True)
    p.add_argument("--input_features_csv", required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--mounted_input_dir", default="/opt/ml/processing/input")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    args = p.parse_args()

    input_file = find_input_csv(args.mounted_input_dir)
    df = pd.read_csv(input_file)

    targets = [c.strip() for c in args.targets_csv.split(",") if c.strip()]
    input_feats = [c.strip() for c in args.input_features_csv.split(",") if c.strip()]

    for tgt in targets:
        tr, va, num_class = per_target_split(df, tgt, input_feats, val_frac=args.val_frac)
        
        out_tr = os.path.join(args.output_dir, tgt, "train")
        out_va = os.path.join(args.output_dir, tgt, "validation")
        out_meta = os.path.join(args.output_dir, tgt, "meta")
        os.makedirs(out_tr, exist_ok=True)
        os.makedirs(out_va, exist_ok=True)
        os.makedirs(out_meta, exist_ok=True)

        tr.to_csv(os.path.join(out_tr, "train.csv"), index=False)
        va.to_csv(os.path.join(out_va, "validation.csv"), index=False)
        with open(os.path.join(out_meta, "class_count.json"), "w") as f:
            json.dump({"num_class": num_class}, f)

if __name__ == "__main__":
    main()
"""

with open("prepare_per_target_splits.py", "w") as f:
    f.write(preprocess_script)
print("✅ Wrote prepare_per_target_splits.py")


# ============================
# Cell 3: Pipeline definition — Preprocess + Train + Deploy (Dynamic num_class)
# ============================
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterBoolean
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
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

split_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

processing_outputs = []
branches = []
model_s3_uris = {}

for tgt in TARGET_COLS:
    processing_outputs.extend([
        ProcessingOutput(output_name=f"train_{tgt}", source=f"/opt/ml/processing/output/{tgt}/train"),
        ProcessingOutput(output_name=f"validation_{tgt}", source=f"/opt/ml/processing/output/{tgt}/validation"),
        ProcessingOutput(output_name=f"meta_{tgt}", source=f"/opt/ml/processing/output/{tgt}/meta")
    ])

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=split_processor,
    inputs=[ProcessingInput(source=input_csv_param, destination="/opt/ml/processing/input")],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", str(VAL_FRAC_DEFAULT),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output"
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d")
)

for tgt in TARGET_COLS:
    train_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"train_{tgt}"].S3Output.S3Uri
    val_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"validation_{tgt}"].S3Output.S3Uri
    meta_s3 = split_step.properties.ProcessingOutputConfig.Outputs[f"meta_{tgt}"].S3Output.S3Uri

    # Dynamically extract num_class
    from urllib.parse import urlparse
    meta_uri = meta_s3.to_string() + "/class_count.json"
    parsed = urlparse(meta_uri)
    meta_bucket = parsed.netloc
    meta_key = parsed.path.lstrip("/")
    import json
    num_class = json.loads(s3.get_object(Bucket=meta_bucket, Key=meta_key)['Body'].read())['num_class']

    xgb = Estimator(
        image_uri=XGB_IMAGE,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=p_sess,
        hyperparameters={"objective": "multi:softprob", "num_class": num_class, "num_round": 50},
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
