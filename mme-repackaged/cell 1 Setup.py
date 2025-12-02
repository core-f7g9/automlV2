# =========================
# Cell 1: Setup
# =========================

import os
import boto3
import sagemaker

region = boto3.Session().region_name
sm_sess = sagemaker.Session()
role = sagemaker.get_execution_role()

BUCKET = sm_sess.default_bucket()

# Input CSV
INPUT_S3 = f"s3://{BUCKET}/input/data.csv"

# Project
CLIENT_NAME = "client1"
PROJECT_NAME = f"{CLIENT_NAME}-manual-ml"
OUTPUT_PREFIX = "manual-ml"

DATA_PREFIX = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

# Targets (one model per target)
TARGET_COLS = [
    "DepartmentCode",
    "AccountCode",
    "SubAccountCode",
    "LocationCode"
]

# Features from your dataset
INPUT_FEATURES = [
    "VendorName",
    "LineDescription",
    "ClubNumber"
]

print("Region:", region)
print("Role:", role)
print("Bucket:", BUCKET)
print("Input:", INPUT_S3)
print("Project:", PROJECT_NAME)
