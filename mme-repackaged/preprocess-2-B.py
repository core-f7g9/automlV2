# multi-target-pipeline/preprocess.py (append after existing block)

# ============================================================
# Cell 2b: Write repack script for MME inference
# ============================================================
import textwrap, os

repack_script = textwrap.dedent("""
import argparse
import glob
import os
import shutil
import tarfile
import tempfile
import textwrap
import json

INFERENCE_TEMPLATE = \"\"\"
import csv
import io
import json
import os

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = {feature_list}
TARGET_NAME = "{target_name}"
...
\"\"\"

def build_inference_script(target_name, features):
    return textwrap.dedent(INFERENCE_TEMPLATE).format(
        target_name=target_name,
        feature_list=json.dumps(features)
    )

def find_model_tar(input_dir):
    candidates = sorted(glob.glob(os.path.join(input_dir, "*.tar.gz")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar.gz"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No .tar.gz artifacts found under {input_dir}")
    return candidates[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input/model")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output/repacked")
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--feature_list_csv", type=str, required=True)
    parser.add_argument("--inference_filename", type=str, default="inference.py")
    args = parser.parse_args()

    features = [c.strip() for c in args.feature_list_csv.split(",") if c.strip()]
    if not features:
        raise ValueError("At least one feature is required for inference")

    script_text = build_inference_script(args.target_name, features)
    src_tar = find_model_tar(args.input_dir)

    work_dir = tempfile.mkdtemp(prefix="repack-")
    try:
        with tarfile.open(src_tar, "r:gz") as tar:
            tar.extractall(work_dir)

        code_dir = os.path.join(work_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        script_path = os.path.join(code_dir, args.inference_filename)
        with open(script_path, "w") as f:
            f.write(script_text)

        os.makedirs(args.output_dir, exist_ok=True)
        dst_tar = os.path.join(args.output_dir, "model.tar.gz")
        with tarfile.open(dst_tar, "w:gz") as tar:
            tar.add(work_dir, arcname=".")

        print(f"Repacked model for target {args.target_name} -> {dst_tar}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
""").strip()

with open("repack_for_mme.py", "w") as f:
    f.write(repack_script)

print("Wrote repack_for_mme.py")
