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
role_arn = sagemaker.get_execution_role()

# ---- Client/project knobs
CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-mme-xgb"
OUTPUT_PREFIX = "mlops"

# ---- Data locations
BUCKET       = sm_sess.default_bucket()
INPUT_S3CSV  = f"s3://{BUCKET}/input/data.csv"
DATA_PREFIX  = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

# ---- Targets
TARGET_COLS = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]

# ---- Features to keep
INPUT_FEATURES = ["VendorName", "LineDescription", "ClubNumber"]

# ---- Train/Val split knobs
VAL_FRAC_DEFAULT        = 0.20
MIN_SUPPORT_DEFAULT     = 5
RARE_TRAIN_ONLY_DEFAULT = True

def _parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {uri}")
    return bucket, key

# Validate CSV exists
s3 = boto3.client("s3", region_name=region)
csv_bucket, csv_key = _parse_s3(INPUT_S3CSV)
try:
    s3.head_object(Bucket=csv_bucket, Key=csv_key)
except ClientError as e:
    raise FileNotFoundError(f"CSV not found at {INPUT_S3CSV}") from e

print("Region:", region)
print("Role:", role_arn)
print("Input CSV:", INPUT_S3CSV)
print("Targets:", TARGET_COLS)
print("Features:", INPUT_FEATURES)
