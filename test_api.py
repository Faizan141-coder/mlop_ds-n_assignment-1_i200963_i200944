import requests


def test_prediction_endpoint():
    url = 'http://127.0.0.1:5000/predict'  # Update the URL if needed
    data = {
        "age": 50,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1
    }
    response = requests.post(url, json=data)
    prediction = response.json()['prediction']
    if prediction == 'Person Not Having Heart Disease':
        print("Test case passed: Person Not Having Heart Disease")
    else:
        # flake8: noqa
        print(f"Test case failed: Prediction was {prediction}, expected Person Not Having Heart Disease")


if __name__ == '__main__':
    test_prediction_endpoint()
