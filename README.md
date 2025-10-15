# Multi-Tenant ML Workflow on Amazon SageMaker Autopilot (AutoML V2)

This repository is a simple, reusable **template** for building multi-tenant ML workflows using **Amazon SageMaker Autopilot V2**.  
Each notebook walks through one stage of the pipeline — from single-tenant setup to multi-tenant automation.

## Contents
- `notebooks/step1_single_tenant.ipynb` — Train and deploy a single-tenant Autopilot V2 model.
- `notebooks/step2_monitoring.ipynb` — Enable data capture and Model Monitor.
- `notebooks/step3_parameterize.ipynb` — Make the workflow configurable and idempotent.
- `notebooks/step4_multitenant.ipynb` — Extend to multiple tenants.
- `notebooks/step5_scheduler.ipynb` — Automate retraining with EventBridge.

## Requirements
- AWS account with SageMaker access
- SageMaker Studio or local Jupyter environment
- `boto3`, `sagemaker`

## Quick start
1. Open SageMaker Studio or local JupyterLab.
2. Clone this repo or click **“Use this template”**.
3. Update S3 paths, IAM Role ARN, and target column in Step 1.
4. Run notebooks sequentially (1 → 5).

## Cleanup
Delete SageMaker endpoints, models, and schedules after testing to avoid ongoing costs.

## License
Apache 2.0