# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def encode_categorical(df):
    le = LabelEncoder()
    for column in df.select_dtypes(include='object').columns:
        df[column] = le.fit_transform(df[column])
    return df
