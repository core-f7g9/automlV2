import boto3
sm = boto3.client("sagemaker", region_name=region)

pkg_group = "client1-autopilot-v2-pkg-group"
pkgs = sm.list_model_packages(ModelPackageGroupName=pkg_group, ModelApprovalStatus="Approved",
                              SortBy="CreationTime", SortOrder="Descending")["ModelPackageSummaryList"]
latest_pkg = pkgs[0]["ModelPackageArn"]

model_name = f"client1-apv2-model"
sm.create_model(ModelName=model_name,
                PrimaryContainer={"ModelPackageName": latest_pkg},
                ExecutionRoleArn=role_arn)

endpoint_name = "client1-apv2-endpoint"
sm.create_endpoint_config(
    EndpointConfigName=endpoint_name + "-cfg",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": "ml.m5.large",
        "InitialInstanceCount": 1
    }]
)
sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_name + "-cfg")
print("Creating endpoint:", endpoint_name)
