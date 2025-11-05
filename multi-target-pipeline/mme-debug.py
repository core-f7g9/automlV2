# ==== Quick test helpers for SageMaker runtime inference ====
import json, boto3, io, csv

smr = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = "client1-autopilot-v1-codes-mme"  # <-- set yours
# If you deployed separate endpoints per target, set ENDPOINT_NAME to that name and call without target_model.

def _csv_line(values):
    """
    Build a single CSV line with correct quoting (no header).
    values: list of Python values in the exact training order.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="")
    writer.writerow(values)
    return buf.getvalue()

def predict_row(endpoint_name, features, target_model=None, content_type="text/csv", accept="application/json"):
    """
    features: list in EXACT order used for training, e.g.
              ["ACME LLC", "12-pack sparkling water, lemon", 42]
    target_model: e.g. "DepartmentCode.tar.gz" for MME; leave None for single-model endpoints
    Returns: decoded string (JSON or plain text) from the model
    """
    payload = _csv_line(features)
    kwargs = {
        "EndpointName": endpoint_name,
        "ContentType": content_type,
        "Accept": accept,
        "Body": payload.encode("utf-8"),
    }
    if target_model:  # MME
        kwargs["TargetModel"] = target_model

    resp = smr.invoke_endpoint(**kwargs)
    body = resp["Body"].read().decode("utf-8", errors="replace")
    # Try to pretty-print JSON if possible
    try:
        return json.dumps(json.loads(body), indent=2)
    except Exception:
        return body

# ===== EXAMPLES =====
# Your feature order must match INPUT_FEATURES = ["VendorName","LineDescription","ClubNumber"]
sample = ["ACME LLC", "12-pack sparkling water, lemon", 42]

# A) MME: pick one of the models placed under the MME prefix (from your pipeline it's <Target>.tar.gz)
print("MME / DepartmentCode prediction:")
print(predict_row(ENDPOINT_NAME, sample, target_model="DepartmentCode.tar.gz"))

# You can test others similarly:
# print(predict_row(ENDPOINT_NAME, sample, target_model="AccountCode.tar.gz"))
# print(predict_row(ENDPOINT_NAME, sample, target_model="SubAccountCode.tar.gz"))
# print(predict_row(ENDPOINT_NAME, sample, target_model="LocationCode.tar.gz"))

# B) Single-model endpoint (if you deployed one model per endpoint):
# ENDPOINT_NAME = "client1-departmentcode-endpoint"  # example
# print(predict_row(ENDPOINT_NAME, sample))
