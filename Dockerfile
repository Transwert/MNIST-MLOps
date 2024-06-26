# docker build --platform linux/amd64 -t ml:0.1.0 .
# docker run -p8000:8000 --shm-size=500mb ml:0.1.0
# Or 
# docker run -p8000:8000 ml:0.1.0

# to stop all containers: docker rm $(docker stop $(docker ps -aq))
# to remove docker image: docker rmi ml --force

# Use a base image with your preferred operating system and dependencies
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --compile -r requirements.txt

# Expose the port your API will run on
EXPOSE 8000

# Define the command to start your API
CMD ["python3", "model_inference.py"]