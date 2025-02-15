# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src/ ./src/
COPY model/ ./model/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r src/requirements.txt

# Run train.py when the container launches
CMD ["python", "src/train.py"]
