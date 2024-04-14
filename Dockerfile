# Use an official Python runtime as a parent image
FROM python:3.8-alpine

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apk add --no-cache gcc musl-dev && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt && \
    apk del gcc musl-dev
RUN apt-get update && apt-get install -y libgomp1

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "Api.py"]
