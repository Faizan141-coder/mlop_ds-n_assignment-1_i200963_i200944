from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained modell
model = joblib.load('heart_failure_pred_model.pkl')


# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    # Convert data into numpy array
    data_array = np.array(list(data.values())).reshape(1, -1)
    # Make prediction
    prediction = model.predict(data_array)
    # Return the prediction
    if prediction[0] == 0:
        result = 'Person Not Having Heart Disease'
    else:
        result = 'Person Having Heart Disease'
    return jsonify({'prediction': result})


# Define a welcome message
@app.route('/')
def welcome():
    return 'Welcome to Heart Disease Prediction API!'


if __name__ == '__main__':
    app.run(debug=True)
