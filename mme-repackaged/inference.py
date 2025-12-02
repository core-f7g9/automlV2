
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import feature_encoder  # noqa: F401 ensures pickled FE loads

def model_fn(model_dir):
    fe = joblib.load(os.path.join(model_dir, "fe.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    bst = xgb.Booster()
    bst.load_model(os.path.join(model_dir, "model.xgb"))
    return fe, bst, label_encoder


def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    return pd.DataFrame(data)


def predict_fn(input_data, model):
    fe, bst, label_encoder = model
    X = fe.transform(input_data)
    dmat = xgb.DMatrix(X)
    preds = bst.predict(dmat)

    classes = label_encoder.classes_.tolist()
    outputs = []
    for row in preds:
        probas = {cls: float(prob) for cls, prob in zip(classes, row)}
        best_idx = int(np.argmax(row))
        outputs.append(
            {
                "predicted_label": classes[best_idx],
                "probabilities": probas,
            }
        )
    return {"predictions": outputs}


def output_fn(prediction, content_type):
    return json.dumps(prediction)
