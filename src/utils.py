# src/utils.py

import pandas as pd
from sklearn.model_selection import train_test_split

def split_features_labels(df, target_column="y"):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
