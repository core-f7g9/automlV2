
import argparse
import os
import shutil
import tarfile
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Directory containing model.tar.gz")
    p.add_argument("--output_tar", required=True, help="Path to write MME-ready tar.gz")
    args = p.parse_args()

    model_tar_path = os.path.join(args.model_dir, "model.tar.gz")
    if not os.path.exists(model_tar_path):
        raise FileNotFoundError(f"Expected model.tar.gz in {args.model_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(tmp)

        inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
        if os.path.exists(inference_src):
            shutil.copy(inference_src, os.path.join(tmp, "inference.py"))

        feature_encoder_src = os.path.join(os.path.dirname(__file__), "feature_encoder.py")
        if os.path.exists(feature_encoder_src):
            shutil.copy(feature_encoder_src, os.path.join(tmp, "feature_encoder.py"))

        with tarfile.open(args.output_tar, "w:gz") as tar:
            tar.add(tmp, arcname=".")


if __name__ == "__main__":
    main()
