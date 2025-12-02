# ============================
# Write inference.py
# ============================

import textwrap, os

inference_script = textwrap.dedent("""
import json
import joblib
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def model_fn(model_dir):
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "xgb-model.json"))

    tfidf_word = joblib.load(os.path.join(model_dir, "tfidf_word.pkl"))
    tfidf_char = joblib.load(os.path.join(model_dir, "tfidf_char.pkl"))
    le         = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    config     = json.load(open(os.path.join(model_dir, "config.json")))

    return {
        "model": booster,
        "tfidf_word": tfidf_word,
        "tfidf_char": tfidf_char,
        "label_encoder": le,
        "config": config
    }

def input_fn(request_body, content_type):
    if content_type == "application/json":
        obj = json.loads(request_body)
        if isinstance(obj, dict):
            return [obj]
        return obj
    raise ValueError("Only JSON supported")

def predict_fn(data, bundle):
    feats = bundle["config"]["text_cols"]

    # Convert row dicts → text feature
    def combine(d):
        return " ".join(str(d.get(c, "")) for c in feats)

    texts = [combine(x) for x in data]

    Xw = bundle["tfidf_word"].transform(texts)
    Xc = bundle["tfidf_char"].transform(texts)

    X = sparse.hstack([Xw, Xc]).tocsr()
    dm = xgb.DMatrix(X)

    preds = bundle["model"].predict(dm)
    labels = preds.argmax(axis=1)
    decoded = bundle["label_encoder"].inverse_transform(labels)

    return decoded.tolist()

def output_fn(prediction, accept):
    return json.dumps({"prediction": prediction})
""")

os.makedirs("xgb_src", exist_ok=True)
with open("xgb_src/inference.py", "w") as f:
    f.write(inference_script)

print("Wrote xgb_src/inference.py")
