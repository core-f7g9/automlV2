# === SageMaker MME End-to-End Debug + Invoke (copy-paste and run) ===
import boto3, io, csv, json, time
from urllib.parse import urlparse

ENDPOINT = "client1-autopilot-v1-codes-mme"   # <-- Your endpoint name (with hyphens)

# Put a simple, comma-safe sample first (no commas in text) to rule out CSV parsing issues.
# Feature order MUST match training: ["VendorName","LineDescription","ClubNumber"]
SAMPLE_FEATURES = ["ACME LLC", "12 pack lemon water", 42]

sm   = boto3.client("sagemaker")
smr  = boto3.client("sagemaker-runtime")
s3   = boto3.client("s3")
logs = boto3.client("logs")

def csv_line(vals, safe=False):
    """
    Build one CSV line (no header).
    If safe=True, avoid quotes/commas to bypass container CSV quirks on first try.
    """
    if safe:
        def q(v):
            s = str(v)
            if ("," in s) or ('"' in s) or ("\n" in s):
                s = s.replace('"', '""')
                return f'"{s}"'
            return s
        return ",".join(q(v) for v in vals)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(vals)
    return buf.getvalue()

def describe_endpoint_details(ep):
    d   = sm.describe_endpoint(EndpointName=ep)
    cfg = sm.describe_endpoint_config(EndpointConfigName=d["EndpointConfigName"])
    mdl_name = cfg["ProductionVariants"][0]["ModelName"]
    md  = sm.describe_model(ModelName=mdl_name)
    cont = md["PrimaryContainer"]
    return {
        "EndpointStatus": d["EndpointStatus"],
        "EndpointConfigName": d["EndpointConfigName"],
        "ModelName": mdl_name,
        "ExecutionRoleArn": md["ExecutionRoleArn"],
        "Image": cont["Image"],
        "ModelDataUrl": cont["ModelDataUrl"],  # for MME this is an S3 PREFIX (ends with /)
    }

def list_mme_models(model_data_url):
    up = urlparse(model_data_url)
    bucket, prefix = up.netloc, up.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    keys, token = [], None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token: kwargs["ContinuationToken"] = token
        r = s3.list_objects_v2(**kwargs)
        keys.extend([o["Key"] for o in r.get("Contents", [])])
        token = r.get("NextContinuationToken")
        if not token: break
    # Relative names under the MME prefix are what TargetModel must be
    rel = [k[len(prefix):] for k in keys if k.endswith(".tar.gz")]
    return bucket, prefix, sorted(rel)

def invoke(ep, target_model, features, safe_csv=True, content_type="text/csv", accept="application/json"):
    body = csv_line(features, safe=safe_csv).encode("utf-8")
    r = smr.invoke_endpoint(
        EndpointName=ep,
        ContentType=content_type,
        Accept=accept,
        Body=body,
        TargetModel=target_model,
    )
    out = r["Body"].read().decode("utf-8", errors="replace")
    try:
        return True, json.dumps(json.loads(out), indent=2)
    except Exception:
        return True, out

def try_invoke(ep, target_model, features):
    try:
        ok, out = invoke(ep, target_model, features, safe_csv=True)
        return ok, out
    except Exception as e1:
        # Retry with strict CSV quoting (handles commas/quotes)
        try:
            ok, out = invoke(ep, target_model, features, safe_csv=False)
            return ok, out
        except Exception as e2:
            return False, f"{type(e2).__name__}: {e2}"

def latest_log_stream(ep):
    group = f"/aws/sagemaker/Endpoints/{ep}"
    r = logs.describe_log_streams(
        logGroupName=group,
        orderBy="LastEventTime",
        descending=True,
        limit=1
    )
    if not r.get("logStreams"):
        return group, None
    return group, r["logStreams"][0]["logStreamName"]

def tail_logs(ep, limit=120):
    group, stream = latest_log_stream(ep)
    if not stream:
        return f"(No CloudWatch logs found for {group})"
    r = logs.get_log_events(logGroupName=group, logStreamName=stream, startFromHead=False, limit=limit)
    return "\n".join(e["message"] for e in r.get("events", []))

# --- Run ---
print("Region:", boto3.Session().region_name)
info = describe_endpoint_details(ENDPOINT)
print("\n=== Endpoint ===")
print(json.dumps(info, indent=2))

status = info["EndpointStatus"]
if status != "InService":
    raise SystemExit(f"Endpoint status is {status}. Wait until InService, then retry.")

bucket, prefix, models = list_mme_models(info["ModelDataUrl"])
print("\n=== MME Prefix ===")
print("S3:", f"s3://{bucket}/{prefix}")
print("\n=== Available models (use these as TargetModel) ===")
for m in models:
    print(" -", m)
if not models:
    raise SystemExit("No .tar.gz models found under MME prefix. Recheck deployment/copy step.")

# Try invoking each discovered model with a simple CSV first
print("\n=== Invoking each model ===")
for m in models:
    ok, out = try_invoke(ENDPOINT, m, SAMPLE_FEATURES)
    print(f"\n[{m}] success={ok}")
    print(out[:2000])

# Optional: try a more realistic description with a comma (strict CSV will handle it)
REALISTIC_SAMPLE = ["ACME LLC", "12-pack sparkling water, lemon", 42]
m0 = models[0]
print(f"\n=== Extra check (first model, realistic sample) -> {m0} ===")
try:
    ok, out = try_invoke(ENDPOINT, m0, REALISTIC_SAMPLE)
    print(f"success={ok}")
    print(out[:2000])
except Exception as e:
    print("Invoke failed:", e)

print("\n=== Recent endpoint logs (last ~120 lines) ===")
print(tail_logs(ENDPOINT, limit=120))
