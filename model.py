# Importing required libs
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import json

# Load the class labels
with open('models/imagenet-simple-labels.json', 'r') as f:
    classes = json.load(f)

# model = models.vgg16(weights="IMAGENET1K_V1")
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Preparing and pre-processing the image
def preprocess_img(img_stream):
    image = Image.open(img_stream)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Make predictions
def predict_result(img_stream):
    img_tensor = preprocess_img(img_stream)
   
    with torch.no_grad():
        output = model(img_tensor)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)    
    predicted_label = classes[predicted_idx.item()]
    return predicted_label

