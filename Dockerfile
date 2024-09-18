# Start from the official Python base image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory's contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
ENV FLASK_APP=N_layerapp

# Run the application. App Engine sets the $PORT environment variable.

CMD flask run --host=0.0.0.0 --port=$PORT