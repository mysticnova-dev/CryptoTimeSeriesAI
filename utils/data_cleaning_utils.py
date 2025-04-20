import pandas as pd
import numpy as np

class MissingValueCleaner:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def interpolate_missing(self, df):
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns
        missing_numeric_cols = [col for col in numeric_cols if df[col].isnull().any()]
        df[missing_numeric_cols] = df[missing_numeric_cols].interpolate(method='linear')
        return df

    def forward_fill_missing(self, df):
        return df.ffill()

    def backward_fill_missing(self, df):
        return df.bfill()

    def median_fill_missing(self, df):
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().any() and np.issubdtype(df[col].dtype, np.number):
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        return df

    def clean(self, df):
        df = df.copy()
        initial_missing = df.isnull().sum()

        df = self.interpolate_missing(df)
        df = self.forward_fill_missing(df)
        df = self.backward_fill_missing(df)
        df = self.median_fill_missing(df)

        final_missing = df.isnull().sum()

        if self.verbose:
            fixed = initial_missing - final_missing
            fixed = fixed[fixed > 0]
            print("Missing values filled per column:\n", fixed)

        return df

