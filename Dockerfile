# Python image to use
FROM python:3.9.19
 
 
# Set the working directory to /app
WORKDIR /
 
# Copy the requirements file used for dependencies
COPY requirements.txt .
 
# Install the needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy rest of the working directory contents into the container at /app
COPY . .
 
# Run app.py when the container launches
ENTRYPOINT ["python3", "app.py"]