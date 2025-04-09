# app/utils/helper.py

def make_prediction(df, model, encoder, scaler):
    # Encode categorical columns
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan']
    df_encoded = df.copy()
    df_encoded[categorical_cols] = encoder.transform(df[categorical_cols])

    # Scale numerical columns
    numerical_cols = ['age', 'balance']
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    # Predict
    prediction = model.predict(df_encoded)
    return "Subscribed" if prediction[0] == 1 else "Not Subscribed"
