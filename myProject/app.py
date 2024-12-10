from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

# Print the current working directory for debugging
print("Current working directory:", os.getcwd())

# Load the trained model using a relative path
model_path = os.path.join("myProject", "student_success_model.h5")
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
else:
    print(f"Model file found at: {model_path}")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Initialize the Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    """API base route"""
    return jsonify({
        "message": "Flask API is running!",
        "routes": ["/predict (POST)"]
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions from input features"""
    print("Request received")  # Debugging log
    data = request.json

    # Check if data contains the 'features' key
    if not data or 'features' not in data:
        print("Invalid input data")  # Debugging log
        return jsonify({"error": "Invalid input. 'features' key is missing."}), 400

    print("Data:", data)  # Debugging log
    try:
        # Convert input features to a NumPy array
        input_features = np.array(data['features']).reshape(1, -1)
        print("Input Features:", input_features)  # Debugging log

        # Predict using the loaded model
        prediction = model.predict(input_features)
        print("Raw Prediction Output:", prediction)  # Debugging log

        # Return prediction result
        result = {"prediction": int(prediction[0][0] > 0.5)}  # Threshold 0.5
        return jsonify(result)
    except Exception as e:
        # Log and return any errors during prediction
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


# Add a custom 404 error page
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 page for invalid routes"""
    return """
    <!doctype html>
    <html lang="en">
    <head>
        <title>404 Not Found</title>
    </head>
    <body>
        <h1>404 - Page Not Found</h1>
        <p>The page you are looking for does not exist. Please check the URL and try again. DALJEET</p>
    </body>
    </html>
    """, 404


if __name__ == '__main__':
    app.run(debug=True)
