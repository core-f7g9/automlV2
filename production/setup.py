# =========================
# Cell 1: Setup variables
# =========================
import os, time, textwrap
import boto3, sagemaker
from urllib.parse import urlparse

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
role_arn = sagemaker.get_execution_role()

# --- EDIT THESE THREE as needed ---
BUCKET      = sm_sess.default_bucket()   # fine to keep; or set to your own bucket name string
INPUT_S3CSV = f"s3://{BUCKET}/input/data.csv"   # <-- point to a real CSV with header
TARGET_COL  = "your_target_col"                 # <-- set to the exact column name in the CSV
DATA_CAPTURE_S3 = f"s3://{BUCKET}/{OUTPUT_PREFIX}/data-capture/"

# Naming knobs (keep consistent with the rest of the notebook)
PROJECT_NAME  = "client1-autopilot-v1"   # use v1 to match Autopilot V1 pipeline
OUTPUT_PREFIX = "mlops"

# --- Early sanity checks to fail fast with clear messages ---
def _parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"INPUT_S3CSV must be an s3:// URI, got: {uri}")
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {uri}")
    return bucket, key

# 1) Check the CSV exists
s3 = boto3.client("s3", region_name=region)
csv_bucket, csv_key = _parse_s3(INPUT_S3CSV)
try:
    s3.head_object(Bucket=csv_bucket, Key=csv_key)
except s3.exceptions.NoSuchKey:
    raise FileNotFoundError(f"CSV not found at {INPUT_S3CSV}. Upload your file or fix the path.")
except Exception as e:
    # Still continue if you prefer, but this helps surface perms/VPC endpoint issues early.
    raise RuntimeError(f"Could not access {INPUT_S3CSV}: {e}")

# 2) Target col must be non-empty
if not TARGET_COL or not isinstance(TARGET_COL, str):
    raise ValueError("TARGET_COL must be a non-empty string matching a column in your CSV header.")

print("Region:", region)
print("Role:", role_arn)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Target:", TARGET_COL)
print("Project:", PROJECT_NAME, "| Output prefix:", OUTPUT_PREFIX)
