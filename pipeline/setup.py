# =========================
# Cell 1: Setup variables
# =========================
import boto3, sagemaker, textwrap, os, time

region   = boto3.Session().region_name
role_arn = sagemaker.get_execution_role()
sm_sess = sagemaker.Session()

# >>> EDIT THESE <<<
BUCKET      = sm_sess.default_bucket()               
INPUT_S3CSV = "s3://your-bucket/path/to/data.csv"   # CSV with header row
TARGET_COL  = "your_target_col"                    

# optional knobs
PROJECT_NAME  = "client1-autopilot-v2"
OUTPUT_PREFIX = "mlops" 

print("Region:", region)
print("Role:", role_arn)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Target:", TARGET_COL)