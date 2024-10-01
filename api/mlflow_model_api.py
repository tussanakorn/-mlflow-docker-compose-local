from flask import Flask, request, jsonify
import mlflow
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Load the model from MLflow
# You can specify the registered model name and version, or use a run ID and model path
model_uri = "models:/MyLogisticRegressionModel/1"  # Adjust version accordingly
model = mlflow.sklearn.load_model(model_uri)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data from the request
        input_data = request.json["data"]
        
        # Ensure the input is in the correct format for prediction
        X = np.array(input_data).reshape(-1, 1)

        # Make predictions using the loaded model
        predictions = model.predict(X)

        # Return predictions as a JSON response
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
