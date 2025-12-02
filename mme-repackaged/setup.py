import os
import boto3
import sagemaker
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger, ParameterFloat, ParameterBoolean
)

region   = boto3.Session().region_name
sm_sess  = sagemaker.Session()
p_sess   = PipelineSession()
role_arn = sagemaker.get_execution_role()

CLIENT_NAME   = "client1"
PROJECT_NAME  = f"{CLIENT_NAME}-automl-xgb"
OUTPUT_PREFIX = "mlops"

BUCKET        = sm_sess.default_bucket()
INPUT_S3CSV   = f"s3://{BUCKET}/input/data.csv"

TARGET_COLS = ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]
INPUT_FEATURES = ["VendorName", "LineDescription", "ClubNumber"]

VAL_FRAC_DEFAULT = 0.20
MIN_SUPPORT_DEFAULT = 5
RARE_TRAIN_ONLY_DEFAULT = True

print("Region:", region)
print("Role:", role_arn)
print("Bucket:", BUCKET)
print("Input CSV:", INPUT_S3CSV)
