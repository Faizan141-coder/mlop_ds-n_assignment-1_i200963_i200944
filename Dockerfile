# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Flask application script and the trained model file into the container
COPY heart_failure_pred.py heart_failure_pred_model.pkl /app/

# Install Flask and other dependencies
RUN pip install Flask numpy scikit-learn joblib

# Expose port 5000 to the outside world
EXPOSE 5000


CMD ["python", "heart_disease_prediction.py"]
