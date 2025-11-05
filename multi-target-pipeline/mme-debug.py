import boto3, io, csv, json
smr = boto3.client("sagemaker-runtime")

ENDPOINT = "<YOUR-ENDPOINT-NAME>"  # e.g., client1-autopilot-v1-codes-mme

def csv_line(vals):
    b = io.StringIO(); csv.writer(b, lineterminator="").writerow(vals); return b.getvalue()

features = ["ACME LLC", "12-pack sparkling water, lemon", 42]  # VendorName, LineDescription, ClubNumber
body = csv_line(features).encode("utf-8")

resp = smr.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="text/csv",
    Accept="application/json",
    Body=body,
    TargetModel="DepartmentCode.tar.gz",  # relative path under the MME S3 prefix
)
print(resp["Body"].read().decode("utf-8"))