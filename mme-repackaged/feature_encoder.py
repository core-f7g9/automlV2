
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical, text, and numeric columns into a numeric matrix."""

    def __init__(self, cat_cols=None, text_cols=None, num_cols=None, max_tfidf_features=5000):
        self.cat_cols = cat_cols or []
        self.text_cols = text_cols or []
        self.num_cols = num_cols or []
        self.max_tfidf_features = max_tfidf_features
        self._cat_encoders = {}
        self._text_vectorizers = {}
        self._scaler = StandardScaler()

    def fit(self, X, y=None):
        for col in self.cat_cols:
            le = LabelEncoder()
            series = X[col].astype(str).fillna("Unknown")
            if "Unknown" not in series.values:
                series = pd.concat([series, pd.Series(["Unknown"])], ignore_index=True)
            le.fit(series)
            self._cat_encoders[col] = le

        for col in self.text_cols:
            tfidf = TfidfVectorizer(max_features=self.max_tfidf_features)
            tfidf.fit(X[col].astype(str).fillna(""))
            self._text_vectorizers[col] = tfidf

        if self.num_cols:
            numeric = self._get_numeric_block(X)
            self._scaler.fit(numeric)

        return self

    def transform(self, X):
        blocks = []

        for col in self.cat_cols:
            series = X[col].astype(str).fillna("Unknown")
            series = series.where(series.isin(self._cat_encoders[col].classes_), "Unknown")
            blocks.append(self._cat_encoders[col].transform(series).reshape(-1, 1))

        for col in self.text_cols:
            series = X[col].astype(str).fillna("")
            vec = self._text_vectorizers[col].transform(series).toarray()
            blocks.append(vec)

        if self.num_cols:
            numeric = self._get_numeric_block(X)
            blocks.append(self._scaler.transform(numeric))

        if not blocks:
            return np.empty((len(X), 0))

        return np.hstack(blocks)

    def _get_numeric_block(self, X):
        numeric = []
        for col in self.num_cols:
            numeric.append(pd.to_numeric(X[col], errors="coerce").fillna(0).values.reshape(-1, 1))
        return np.hstack(numeric) if numeric else np.empty((len(X), 0))
