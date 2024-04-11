import requests
import json
from utils import imgGen

class client:
    def __init__(self):

        # Sample input data
        self.mainFunc = imgGen()
        # input_data = {
        #     "image": [[0.89]*28]*28  # Example input data
        # }
        self.input_data = {
            "image": self.mainFunc.randDecMatrix()
        }

        # Endpoint URL
        self.url = "http://0.0.0.0:8000/predict"  # Update with your server's address

    def infer(self):
        """
        Main inference function which sends POST Request,
        takes default input i.e. random initialised image
        """
        # Send POST request
        try:
            response = requests.post(self.url, json=self.input_data)
            # Check response status
            if response.status_code == 200:
                result = response.json()
                print("Prediction:", result["prediction"]) #comment this part when running stressTesting.py
                return response
            else:
                print("Error:", response.text)
                return response
        except Exception as e:
            print("Exception happened: ", e)

if __name__ == "__main__":
    run = client()
    run.infer()