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

# ---- S3 sanity check
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
# Cell 2: Write helper scripts
#   - prepare_per_target_splits.py
#   - deploy_xgb_mme.py
# =========================

# --------- prepare_per_target_splits.py (preprocess for XGBoost) ----------
preprocess_script = r"""
import argparse, os, glob, json
import pandas as pd
import numpy as np


def find_input_csv(mounted_dir):
    candidates = glob.glob(os.path.join(mounted_dir, "*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under {mounted_dir}")
    # If multiple, just take first; adjust if you need a specific one
    return candidates[0]


def encode_features(df: pd.DataFrame, input_feats):
    '''
    Ensure all feature columns are numeric.
    - Numeric columns are left as-is.
    - Non-numeric (object/string/category) are converted to category codes.
    This is simple but works fine for tree-based XGBoost.
    '''
    df_feats = df[input_feats].copy()

    for col in input_feats:
        col_data = df_feats[col]
        if pd.api.types.is_numeric_dtype(col_data):
            # already numeric
            continue

        # Convert text / categorical to integer codes
        df_feats[col] = col_data.astype('category').cat.codes.astype('int32')

    return df_feats


def per_target_split(full_df, enc_feats, target_col, input_feats, val_frac=0.2):
    '''
    full_df   : original dataframe (with target column)
    enc_feats : numeric feature dataframe aligned by index with full_df
    target_col: name of target column
    '''
    # Drop rows with missing target
    df_t = full_df[~full_df[target_col].isna()].copy()

    # Build label encoding 0..num_class-1
    classes = sorted(df_t[target_col].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    df_t['label'] = df_t[target_col].astype(str).map(class_to_idx).astype('int32')

    # Align encoded features with this subset
    feat_t = enc_feats.loc[df_t.index].copy()

    # Final matrix: [label] + encoded features
    df_all = pd.concat([df_t['label'], feat_t], axis=1)

    # Stratified split by label
    val_idx = []
    for lbl in sorted(df_all['label'].unique()):
        g = df_all[df_all['label'] == lbl]
        n_val = max(1, int(len(g) * val_frac))
        val_idx.extend(g.sample(n=n_val, random_state=42).index)

    val = df_all.loc[val_idx]
    train = df_all.drop(index=val_idx)

    num_class = len(classes)
    return train, val, num_class, class_to_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets_csv', required=True)
    p.add_argument('--input_features_csv', required=True)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--mounted_input_dir', default='/opt/ml/processing/input')
    p.add_argument('--output_dir', default='/opt/ml/processing/output')
    args = p.parse_args()

    input_file = find_input_csv(args.mounted_input_dir)
    print(f'Using input CSV: {input_file}')
    # Your dataset has header + text + numeric
    df = pd.read_csv(input_file)

    targets = [c.strip() for c in args.targets_csv.split(',') if c.strip()]
    input_feats = [c.strip() for c in args.input_features_csv.split(',') if c.strip()]

    # Basic sanity checks
    missing_feats = [c for c in input_feats if c not in df.columns]
    missing_tgts = [c for c in targets if c not in df.columns]
    if missing_feats:
        raise ValueError(f'Missing feature columns in CSV: {missing_feats}')
    if missing_tgts:
        raise ValueError(f'Missing target columns in CSV: {missing_tgts}')

    # Encode ALL features once, globally, so encoding is consistent across targets
    enc_feats = encode_features(df, input_feats)

    for tgt in targets:
        print(f'Processing target: {tgt}')

        train_df, val_df, num_class, class_to_idx = per_target_split(
            full_df=df,
            enc_feats=enc_feats,
            target_col=tgt,
            input_feats=input_feats,
            val_frac=args.val_frac,
        )

        out_tr = os.path.join(args.output_dir, tgt, 'train')
        out_va = os.path.join(args.output_dir, tgt, 'validation')
        out_meta = os.path.join(args.output_dir, tgt, 'meta')

        os.makedirs(out_tr, exist_ok=True)
        os.makedirs(out_va, exist_ok=True)
        os.makedirs(out_meta, exist_ok=True)

        # IMPORTANT: no header, label is first column
        train_path = os.path.join(out_tr, 'train.csv')
        val_path = os.path.join(out_va, 'validation.csv')

        train_df.to_csv(train_path, index=False, header=False)
        val_df.to_csv(val_path, index=False, header=False)

        meta_path = os.path.join(out_meta, 'class_count.json')
        with open(meta_path, 'w') as f:
            json.dump(
                {
                    'num_class': int(num_class),
                    'class_to_idx': class_to_idx,
                },
                f,
            )

        print(
            f'Target {tgt}: wrote '
            f'train -> {train_path} (rows={len(train_df)}), '
            f'val -> {val_path} (rows={len(val_df)}), '
            f'num_class={num_class}'
        )


if __name__ == '__main__':
    main()
"""

with open("prepare_per_target_splits.py", "w") as f:
    f.write(preprocess_script)
print("Wrote prepare_per_target_splits.py")


# --------- deploy_xgb_mme.py (Lambda for multi-model endpoint deploy) ----------
deploy_script = r"""
import os
import time
import json
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def _parse_s3_uri(uri: str):
    if not uri.startswith('s3://'):
        raise ValueError(f'Expected s3:// URI, got: {uri}')
    p = urlparse(uri)
    return p.netloc, p.path.lstrip('/')


def handler(event, context):
    '''
    Lambda entrypoint used by the SageMaker Pipeline LambdaStep.

    Expected event keys (from LambdaStep inputs):
      - EndpointName
      - InstanceType
      - InitialInstanceCount
      - ModelsPrefix        (s3://.../mme/client1/models/)
      - TargetNamesCSV      (e.g. "DepartmentCode,AccountCode,...")
      - TargetModelDatasCSV (CSV of S3 URIs to model.tar.gz from training)
      - XGBoostImage
    '''

    print('Received event:', json.dumps(event))

    endpoint_name = event['EndpointName']
    instance_type = event['InstanceType']
    initial_instance_count = int(event['InitialInstanceCount'])
    models_prefix_uri = event['ModelsPrefix']
    target_names_csv = event['TargetNamesCSV']
    target_models_csv = event['TargetModelDatasCSV']
    image_uri = event['XGBoostImage']

    # Execution role for the SageMaker model
    exec_role_arn = os.environ.get('EXEC_ROLE_ARN')
    if not exec_role_arn:
        raise RuntimeError('EXEC_ROLE_ARN environment variable not set')

    sm = boto3.client('sagemaker')
    s3 = boto3.client('s3')

    target_names = [t.strip() for t in target_names_csv.split(',') if t.strip()]
    model_uris = [m.strip() for m in target_models_csv.split(',') if m.strip()]

    if len(target_names) != len(model_uris):
        raise ValueError(
            f'Mismatch between targets ({len(target_names)}) '
            f'and model URIs ({len(model_uris)})'
        )

    # Parse destination prefix for the MME models
    dest_bucket, dest_prefix = _parse_s3_uri(models_prefix_uri)
    dest_prefix = dest_prefix.rstrip('/') + '/'

    copied_models = []

    # Copy each model artifact into the MME prefix
    for tgt, src_uri in zip(target_names, model_uris):
        src_bucket, src_key = _parse_s3_uri(src_uri)
        # Use a stable filename per target
        dest_key = f'{dest_prefix}{tgt}.tar.gz'

        print(f"Copying model for target '{tgt}' from {src_uri} to s3://{dest_bucket}/{dest_key}")

        copy_source = {'Bucket': src_bucket, 'Key': src_key}
        s3.copy_object(Bucket=dest_bucket, Key=dest_key, CopySource=copy_source)

        copied_models.append(
            {
                'target': tgt,
                'source': src_uri,
                'destination': f's3://{dest_bucket}/{dest_key}',
            }
        )

    # Create or update the SageMaker multi-model endpoint
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_name = f'{endpoint_name}-model-{timestamp}'
    endpoint_config_name = f'{endpoint_name}-config-{timestamp}'

    # Multi-model container definition
    container = {
        'Image': image_uri,
        'Mode': 'MultiModel',
        'ModelDataUrl': models_prefix_uri.rstrip('/'),  # prefix only
        'Environment': {},
    }

    print(f"Creating model '{model_name}' with ModelDataUrl={container['ModelDataUrl']}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container,
        ExecutionRoleArn=exec_role_arn,
    )

    print(
        f"Creating endpoint config '{endpoint_config_name}' "
        f"for endpoint '{endpoint_name}' (type={instance_type}, count={initial_instance_count})"
    )
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': initial_instance_count,
                'InstanceType': instance_type,
                'InitialVariantWeight': 1.0,
            }
        ],
    )

    # Check if endpoint exists
    endpoint_exists = False
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        print(f"Endpoint '{endpoint_name}' exists; will update.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException' or 'Could not find endpoint' in str(e):
            print(f"Endpoint '{endpoint_name}' does not exist; will create.")
            endpoint_exists = False
        else:
            raise

    if endpoint_exists:
        resp = sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = 'updated'
        print('Update endpoint response:', json.dumps(resp, default=str))
    else:
        resp = sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = 'created'
        print('Create endpoint response:', json.dumps(resp, default=str))

    result = {
        'status': f'endpoint_{action}',
        'endpoint_name': endpoint_name,
        'model_name': model_name,
        'endpoint_config_name': endpoint_config_name,
        'models_prefix': models_prefix_uri,
        'copied_models': copied_models,
    }

    print('Result:', json.dumps(result))
    return result
"""

with open("deploy_xgb_mme.py", "w") as f:
    f.write(deploy_script)
print("Wrote deploy_xgb_mme.py")

# ============================
# Cell 3: Pipeline definition — Preprocess + Train + Deploy (dynamic num_class)
# ============================
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.properties import PropertyFile

p_sess = PipelineSession()

bucket_param         = ParameterString("Bucket",       default_value=BUCKET)
input_csv_param      = ParameterString("InputS3CSV",   default_value=INPUT_S3CSV)
EndpointNameParam    = ParameterString("EndpointName", default_value=f"{PROJECT_NAME}-mme")
InstanceTypeParam    = ParameterString("InstanceType", default_value="ml.m5.large")
InitialInstanceCount = ParameterInteger("InitialInstanceCount", default_value=1)

XGB_IMAGE = sagemaker.image_uris.retrieve("xgboost", region, version="1.3-1")

# --- Preprocessing ScriptProcessor ---
split_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    command=["python3"],
    sagemaker_session=p_sess,
)

# ✅ Only ONE ProcessingOutput: the root of /opt/ml/processing/output
processing_outputs = [
    ProcessingOutput(
        output_name="data",
        source="/opt/ml/processing/output",
    )
]

# Per-target metadata, but all under the same output_name="data"
meta_property_files = []
for tgt in TARGET_COLS:
    meta_property_files.append(
        PropertyFile(
            name=f"{tgt}Meta",
            output_name="data",
            path=f"{tgt}/meta/class_count.json",  # nested path inside /output
        )
    )

split_step = ProcessingStep(
    name="PreparePerTargetSplits",
    processor=split_processor,
    inputs=[
        ProcessingInput(
            source=input_csv_param,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=processing_outputs,
    code="prepare_per_target_splits.py",
    job_arguments=[
        "--targets_csv", ",".join(TARGET_COLS),
        "--input_features_csv", ",".join(INPUT_FEATURES),
        "--val_frac", str(VAL_FRAC_DEFAULT),
        "--mounted_input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
    ],
    cache_config=CacheConfig(enable_caching=True, expire_after="7d"),
    property_files=meta_property_files,
)

# S3 root where all processed data lives
data_root_s3 = split_step.properties.ProcessingOutputConfig.Outputs["data"].S3Output.S3Uri

# --- Per-target training steps & model artifact collection ---
train_steps = []
model_s3_uris = {}

for tgt in TARGET_COLS:
    # Build per-target train/val S3 prefixes using Join (no Python string concatenation)
    train_s3 = Join(on="", values=[data_root_s3, tgt, "/train"])
    val_s3   = Join(on="", values=[data_root_s3, tgt, "/validation"])

    # dynamic num_class from preprocessing metadata
    meta_file = next(pf for pf in meta_property_files if pf.name == f"{tgt}Meta")
    num_class = JsonGet(
        step_name=split_step.name,
        property_file=meta_file,
        json_path="num_class",
    )

    xgb_estimator = Estimator(
        image_uri=XGB_IMAGE,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=p_sess,
        hyperparameters={
            "objective": "multi:softprob",
            "num_class": num_class,
            "num_round": 50,
        },
    )

    train_step = TrainingStep(
        name=f"TrainModel_{tgt}",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(val_s3, content_type="text/csv"),
        },
    )

    model_s3_uris[tgt] = train_step.properties.ModelArtifacts.S3ModelArtifacts
    train_steps.append(train_step)

# --- Deployment via Lambda to Multi-Model Endpoint ---
MME_MODELS_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}/mme/{CLIENT_NAME}/models/"

deploy_lambda_name = f"{PROJECT_NAME}-deploy-mme"

deploy_lam = Lambda(
    function_name=deploy_lambda_name,
    execution_role_arn=role,          # Lambda's execution role
    script="deploy_xgb_mme.py",
    handler="deploy_xgb_mme.handler",
    timeout=600,
    memory_size=512,
    environment={
        "Variables": {
            "EXEC_ROLE_ARN": role  # SageMaker model execution role
        }
    },
)

target_names_csv   = ",".join(TARGET_COLS)
target_model_uris  = [model_s3_uris[t] for t in TARGET_COLS]
target_datas_csv   = Join(on=",", values=target_model_uris)

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
        "XGBoostImage": XGB_IMAGE,
    },
    outputs=[
        LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)
    ],
)

pipeline = Pipeline(
    name=f"{PROJECT_NAME}-pipeline",
    parameters=[
        bucket_param,
        input_csv_param,
        EndpointNameParam,
        InstanceTypeParam,
        InitialInstanceCount,
    ],
    steps=[split_step] + train_steps + [deploy_step],
    sagemaker_session=p_sess,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Pipeline execution ARN:", execution.arn)
