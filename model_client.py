import requests
import json
from utils import imgGen


# Sample input data
mainFunc = imgGen()
# input_data = {
#     "image": [[0.89]*28]*28  # Example input data
# }
input_data = {
    "image": mainFunc.randDecMatrix()
}

# Endpoint URL
url = "http://0.0.0.0:8000/predict"  # Update with your server's address

# Send POST request
try:
    response = requests.post(url, json=input_data)
    # Check response status
    if response.status_code == 200:
        result = response.json()
        print("Prediction:", result["prediction"])
    else:
        print("Error:", response.text)
except Exception as e:
    print("Exception happened: ", e)