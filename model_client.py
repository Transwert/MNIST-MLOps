import requests
import json

# Sample input data
input_data = {
    "image": [[0.89]*28]*28  # Example input data
}

# Endpoint URL
url = "http://127.0.0.1:4567/predict/"  # Update with your server's address

# Send POST request
response = requests.post(url, json=input_data)

# Check response status
if response.status_code == 200:
    result = response.json()
    print("Prediction:", result["prediction"])
else:
    print("Error:", response.text)