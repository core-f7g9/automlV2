# =========================
# Cell 1: Setup variables
# =========================
import os, json
import boto3, sagemaker
from urllib.parse import urlparse
from botocore.exceptions import ClientError

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
role_arn = sagemaker.get_execution_role()  # OK in SageMaker Studio

# ---- Client/project knobs
CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-autopilot-v1"
OUTPUT_PREFIX = "mlops"

# ---- Data locations
BUCKET       = sm_sess.default_bucket()                      # or set your own
INPUT_S3CSV  = f"s3://{BUCKET}/input/data.csv"              # must exist; header present
DATA_PREFIX  = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

# ---- Targets (4 total) and inputs (whitelist)
TARGET_COLS      = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]
INPUT_FEATURES   = ["VendorName", "LineDescription", "ClubNumber"]
FEATURE_SPEC_S3  = f"{DATA_PREFIX}/features/features.json"

def _parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {uri}")
    return bucket, key

# ---- Early sanity: CSV must exist
s3 = boto3.client("s3", region_name=region)
csv_bucket, csv_key = _parse_s3(INPUT_S3CSV)
try:
    s3.head_object(Bucket=csv_bucket, Key=csv_key)
except ClientError as e:
    code = e.response.get("Error", {}).get("Code", "")
    if code in ("404", "NoSuchKey", "NotFound"):
        raise FileNotFoundError(f"CSV not found at {INPUT_S3CSV}. Upload your file or fix the path.") from e
    raise RuntimeError(f"Could not access {INPUT_S3CSV}: {e}") from e

# ---- Create & upload Feature Specification whitelist
feature_spec = {"FeatureAttributeNames": INPUT_FEATURES}
fs_bucket, fs_key = _parse_s3(FEATURE_SPEC_S3)
s3.put_object(
    Bucket=fs_bucket,
    Key=fs_key,
    Body=json.dumps(feature_spec).encode("utf-8"),
    ContentType="application/json",
)
print(f"Uploaded feature spec to {FEATURE_SPEC_S3}: {feature_spec}")

print("Region:", region)
print("Role:", role_arn)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Targets (4):", TARGET_COLS)
print("Input features (whitelist):", INPUT_FEATURES)
print("Project:", PROJECT_NAME, "| Output prefix:", OUTPUT_PREFIX)
