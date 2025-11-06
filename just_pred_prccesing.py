from typing import Optional, Sequence, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class BaseDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, needed_columns: Optional[Sequence[str]] = None):
        self.needed_columns = list(needed_columns) if needed_columns is not None else None
        self.scaler = StandardScaler()
        self.columns_: Optional[List[str]] = None

    def fit(self, data: pd.DataFrame, *args):
        if self.needed_columns is None:
            cols = list(data.columns)
        else:
            missing = [c for c in self.needed_columns if c not in data.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            cols = list(self.needed_columns)
        self.columns_ = cols
        self.scaler.fit(data[self.columns_].to_numpy(dtype=float))
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("Call fit before transform.")
        missing = [c for c in self.columns_ if c not in data.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")
        X = data[self.columns_].to_numpy(dtype=float)
        return self.scaler.transform(X)