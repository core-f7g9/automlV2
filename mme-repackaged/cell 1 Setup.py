# ============================
# Cell 1: Setup variables
# ============================

import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
p_sess   = PipelineSession()
role_arn = sagemaker.get_execution_role()

CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-automl-v2-xgb"
OUTPUT_PREFIX = "mlops"

BUCKET        = sm_sess.default_bucket()
INPUT_S3CSV   = f"s3://{BUCKET}/input/data.csv"

TARGET_COLS    = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]
INPUT_FEATURES = ["VendorName", "LineDescription", "ClubNumber"]

print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
print("Role:", role_arn)
print("Region:", region)
