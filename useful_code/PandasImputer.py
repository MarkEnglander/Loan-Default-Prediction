import pandas as pd
from sklearn.impute import SimpleImputer


class PandasImputer(SimpleImputer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)