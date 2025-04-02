# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model file
COPY app.py .
COPY kmeans_model.pkl .

# Expose the application port
EXPOSE 8080

# Run the Flask app using Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

