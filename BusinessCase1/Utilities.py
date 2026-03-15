import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import gower


def normalize_df(df, scaler):
    """
    Normalize only the numerical columns of a DataFrame,
    leaving categorical/object columns untouched so Gower can read them.
    """
    df_norm = df.copy()
    num_cols = df_norm.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df_norm[num_cols] = scaler.fit_transform(df_norm[num_cols])
    return df_norm

def find_outliers_selective(df, columns_to_check):
    """
    df: Your pandas DataFrame
    columns_to_check: List of strings (names of continuous numerical columns)
    """
    rows_to_drop = set()

    for col in columns_to_check:
        data = df[col].values
        mean, std = data.mean(), data.std()

        # 3-sigma rule
        lower, upper = mean - 3* std, mean + 3 * std

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        if not outliers.empty:
            print(f"Feature '{col}' has {len(outliers)} outliers.")
            rows_to_drop.update(outliers.index.tolist())

    # Drop rows by index
    df_cleaned = df.drop(index=list(rows_to_drop))
    print(f"\nDropped {len(rows_to_drop)} total rows.")
    return df_cleaned