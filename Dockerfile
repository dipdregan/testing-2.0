# Use the official Python 3.8 slim image based on Debian Buster
FROM python:3.8-slim-buster

# Update package lists and install the AWS CLI
RUN apt update -y && apt install awscli -y

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Specify the command to run on container startup
CMD ["python3", "app.py"]
