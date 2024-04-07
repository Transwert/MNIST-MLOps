from fastapi import FastAPI
from pydantic import BaseModel, ValidationError, validator
from typing import List

import uvicorn
# import pickle
from model_train import Net

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Create the app
app = FastAPI()

class MNISTInput(BaseModel):
    image: List[List[float]]

    class Config:
        schema_extra = {
            "example": {
                "image": [[0.0]*28]*28  # Example input data
            }
        }

def to_tensor(self):
    # Convert image data to torch tensor
    tensor = torch.tensor(self.image).unsqueeze(0).unsqueeze(0).float()
    return tensor

# Load trained Pipeline
network = Net()
network.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')))
network.eval()

# Function to perform inference on image
def predict_image(image):
    # Define the transformations for input data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    outputs = network(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Define predict function
@app.post('/predict')
def predict(item: MNISTInput):    
    predictions = {}
    try:
        # Convert input data to tensor
        image_tensor = to_tensor(item)

        # Perform inference
        with torch.no_grad():
            outputs = network(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        # Return prediction result
        print({"prediction": predicted.item()})
        return {"prediction": predicted.item()}
    except Exception as e:
        print("Exception happened: ", e)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4567)
