# Use an official Python runtime as a parent image

FROM python:3.9.13-slim



# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Met à jour les packages et installe libgomp
RUN apt-get update && apt-get install -y libgomp1 libomp-dev  # Adapt for other systems

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "Api.py"]