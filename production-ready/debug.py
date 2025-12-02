import tarfile
import boto3

def check_model_for_target(tgt):
    group = f"client1-{tgt}-models"
    sm = boto3.client("sagemaker")
    
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]
    details = sm.describe_model_package(ModelPackageName=arn)
    uri = details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    bucket, key = uri.replace("s3://", "").split("/", 1)
    local = f"{tgt}.tar.gz"
    boto3.client("s3").download_file(bucket, key, local)

    print(f"---- {tgt} ----")
    with tarfile.open(local, "r:gz") as tar:
        print(tar.getnames())

for tgt in ["DepartmentCode", "AccountCode", "SubAccountCode", "LocationCode"]:
    check_model_for_target(tgt)
