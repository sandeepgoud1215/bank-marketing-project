import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def encode_and_scale(df):
    df = df.copy()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, le, scaler
