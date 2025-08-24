# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model artifacts into the container
COPY main.py .
COPY heart_disease_model.joblib .
COPY imputer.joblib .
COPY label_encoder.joblib .
COPY training_columns.joblib .
# Expose port 80 for the application
EXPOSE 80

# Run main.py when the container launches
# Use 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
