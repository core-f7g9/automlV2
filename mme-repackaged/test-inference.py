# === Quick MME invoke smoke test ===
import boto3, csv, io, json

# Fill these in
ENDPOINT_NAME = "client1-autopilot-v1-codes-mme"   # your deployed endpoint
TARGET_MODEL  = "DepartmentCode.tar.gz"            # one of the .tar.gz files in the MME prefix
FEATURES      = ["ACME LLC", "12 pack lemon water", 42]  # in the same order as INPUT_FEATURES

def csv_line(row):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(row)
    return buf.getvalue()

smr = boto3.client("sagemaker-runtime")

resp = smr.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="text/csv",
    TargetModel=TARGET_MODEL,
    Body=csv_line(FEATURES),
)

body = resp["Body"].read().decode("utf-8", errors="replace")
try:
    print(json.dumps(json.loads(body), indent=2))
except Exception:
    print(body)
