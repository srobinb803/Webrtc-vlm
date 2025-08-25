# Dockerfile for modern Docker versions

# Use a specific, recent version of Python for reproducibility.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies needed by OpenCV in a robust and clean way.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install Python dependencies.
# Using a virtual environment inside the container is a good practice,
# but for simplicity in this project, we install globally.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container.
# The .dockerignore file will prevent unnecessary files from being copied.
COPY . .

# Expose the port the server will run on.
EXPOSE 8080

# Define the command to run the application using the recommended JSON array format.
CMD ["python", "server.py"]