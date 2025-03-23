import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Load the saved model
model = tf.keras.models.load_model('sniper_model.keras.h5')


import joblib

scaler = joblib.load('scaler.pkl')  # Load the scaler

# Function to preprocess the input data and make predictions
def predict_hit_or_miss(input_data):
    try:
        # Convert the input data string into a numpy array of floats
        input_values = np.array([list(map(float, input_data.split(',')))])

        # Ensure the correct number of features is passed in
        if input_values.shape[1] != 7:  # Adjust 7 to the number of features in your model
            raise ValueError(f"Expected 7 features, but got {input_values.shape[1]}.")

        # Normalize the input using the same scaler used during training
        input_scaled = scaler.transform(input_values)

        # Predict using the model
        prediction = model.predict(input_scaled)

        # Convert the prediction to binary class (0 = Miss, 1 = Hit)
        predicted_class = (prediction > 0.5).astype(int)

        # Print the predicted probability and class
        print(f"Predicted Probability: {prediction[0][0]:.2f}")
        print(f"Predicted Class: {'Hit' if predicted_class[0][0] == 1 else 'Miss'}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Example: You can copy-paste your input data as a comma-separated string
input_data = "51.873,1.479471,76.0825,91.10352,-1.731583,1.479471,51.82298"  # New input data as string

# Call the prediction function
predict_hit_or_miss(input_data)
