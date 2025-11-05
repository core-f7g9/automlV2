import boto3, io, csv, json, time
from urllib.parse import urlparse

ENDPOINT = "<YOUR-ENDPOINT-NAME>"   # e.g., client1-autopilot-v1-codes-mme
# Your features in EXACT training order:
FEATURES = ["ACME LLC", "12 pack lemon water", 42]  # use a description WITHOUT commas first to rule out CSV parsing

sm = boto3.client("sagemaker")
smr = boto3.client("sagemaker-runtime")
s3  = boto3.client("s3")
logs = boto3.client("logs")

def csv_line(vals, safe=False):
    """
    If safe=True, builds a 'bare' CSV line without quotes where possible, to avoid container CSV quirks.
    Start with safe=True to avoid commas/quotes. If that works, switch back to proper quoting.
    """
    if safe:
        # crude but effective: quote only if needed
        def q(v):
            s = str(v)
            return s if ("," not in s and '"' not in s and "\n" not in s) else f'"{s.replace("\"","\"\"")}"'
        return ",".join(q(v) for v in vals)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(vals)
    return buf.getvalue()

def describe_endpoint_details(ep):
    d = sm.describe_endpoint(EndpointName=ep)
    cfg = sm.describe_endpoint_config(EndpointConfigName=d["EndpointConfigName"])
    model_name = cfg["ProductionVariants"][0]["ModelName"]
    md = sm.describe_model(ModelName=model_name)
    container = md["PrimaryContainer"]
    role = md["ExecutionRoleArn"]
    model_data_url = container["ModelDataUrl"]  # should be an S3 PREFIX for MME
    image = container["Image"]
    return {
        "EndpointStatus": d["EndpointStatus"],
        "ModelName": model_name,
        "ExecutionRoleArn": role,
        "ModelDataUrl": model_data_url,
        "Image": image,
    }

def list_mme_objects(model_data_url):
    up = urlparse(model_data_url)
    bucket, prefix = up.netloc, up.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    objs = []
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token: kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        objs += [o["Key"] for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if not token: break
    # Return basenames relative to the prefix (what TargetModel must be)
    rel = []
    for k in objs:
        if k.endswith(".tar.gz"):
            rel.append(k[len(prefix):])
    return bucket, prefix, sorted(rel)

def try_invoke(ep, target_model, features, content_type="text/csv", accept="application/json", safe_csv=True):
    body = csv_line(features, safe=safe_csv).encode("utf-8")
    try:
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
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def latest_log_stream_name(ep):
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

def tail_endpoint_logs(ep, limit=100):
    group, stream = latest_log_stream_name(ep)
    if not stream:
        return f"(No logs found for {group})"
    r = logs.get_log_events(logGroupName=group, logStreamName=stream, startFromHead=False, limit=limit)
    lines = [e["message"] for e in r.get("events", [])]
    return "\n".join(lines[-limit:])

# ---- Run checks ----
info = describe_endpoint_details(ENDPOINT)
print("Endpoint:", ENDPOINT)
print(json.dumps(info, indent=2))

bucket, prefix, models = list_mme_objects(info["ModelDataUrl"])
print("\nMME S3 prefix:", f"s3://{bucket}/{prefix}")
print("Discovered model files (relative names to use as TargetModel):")
for m in models:
    print("  -", m)

if not models:
    print("\n!! No .tar.gz models found under the MME prefix. Deployment likely didn’t copy artifacts.")
else:
    print("\n=== Invoking each discovered model (safe CSV first) ===")
    for m in models:
        ok, out = try_invoke(ENDPOINT, m, FEATURES, safe_csv=True)
        print(f"\n[{m}] success={ok}")
        print(out[:2000])

    print("\n=== If all failed, retry first model with strict CSV quoting ===")
    m0 = models[0]
    ok, out = try_invoke(ENDPOINT, m0, FEATURES, safe_csv=False)
    print(f"\n[{m0} strict CSV] success={ok}")
    print(out[:2000])

print("\n=== Recent endpoint logs (last ~100 lines) ===")
print(tail_endpoint_logs(ENDPOINT, limit=120))
