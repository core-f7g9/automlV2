# ============================================
# (Optional) Cell 4: Deploy the registered model
# ============================================
# You can run this after the pipeline completes successfully.
import boto3
sm = boto3.client("sagemaker", region_name=region)

pkg_group = f"{PROJECT_NAME}-pkg-group"
pkgs = sm.list_model_packages(
    ModelPackageGroupName=pkg_group,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending"
)["ModelPackageSummaryList"]

latest_pkg = pkgs[0]["ModelPackageArn"]

model_name = f"{PROJECT_NAME}-model"
endpoint_name = f"{PROJECT_NAME}-endpoint"

# Create model from the approved package
sm.create_model(
    ModelName=model_name,
    PrimaryContainer={"ModelPackageName": latest_pkg},
    ExecutionRoleArn=role_arn
)

# Create endpoint config & endpoint (single dedicated instance for now)
sm.create_endpoint_config(
    EndpointConfigName=endpoint_name + "-cfg",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": "ml.m5.large",
        "InitialInstanceCount": 1
    }]
)

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_name + "-cfg"
)

print("Creating endpoint:", endpoint_name)