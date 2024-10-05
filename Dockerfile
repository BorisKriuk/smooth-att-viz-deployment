# # Use an official Python runtime as the base image
# FROM python:3.8-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Make port 8000 available to the world outside this container
# EXPOSE 8000

# # Define environment variable for Flask
# ENV FLASK_APP=app.py

# # Run app.py when the container launches
# CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY app.py .
COPY model.pth .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for the model path
ENV MODEL_PATH=/app/model.pth

# Run gunicorn when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]