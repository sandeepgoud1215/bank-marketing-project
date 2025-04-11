# main.py

import pandas as pd
from src.preprocessing import preprocess_data
from src.training import train_model, save_model, save_object

# Step 1: Load the dataset
df = pd.read_csv("bankmarketing.csv")

# Step 2: Preprocess the data
X_scaled, y, encoder, scaler = preprocess_data(df)

# Step 3: Train the model
model = train_model(X_scaled, y)

# Step 4: Save model, encoder, and scaler
save_model(model, filename="model.pkl")
save_object(encoder, filename="encoder.pkl")
save_object(scaler, filename="scaler.pkl")

print("✅ Model, encoder, and scaler saved successfully!")
