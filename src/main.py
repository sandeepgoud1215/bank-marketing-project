from src.training import save_object

# Save encoder and scaler
save_object(encoder, "encoder.pkl")
save_object(scaler, "scaler.pkl")

print("Encoder and Scaler saved as encoder.pkl and scaler.pkl")


# Step 6: Save the model and preprocessing objects
save_model(model, filename="model.pkl")
save_object(encoder, "encoder.pkl")
save_object(scaler, "scaler.pkl")

print("Model, Encoder, and Scaler saved successfully.")
