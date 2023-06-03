# Importing required libs
import numpy as np
from PIL import Image

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications import MobileNet


import torch
import torchvision.models as models
import torchvision.transforms as transforms

import json

# Load the class labels
with open('models/imagenet-simple-labels.json', 'r') as f:
    classes = json.load(f)

model = models.vgg16(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Loading model
# model = load_model('models/digit_model.h5')
# model = MobileNet(weights='imagenet')


# Preparing and pre-processing the image

def preprocess_img(img_stream):
    image = Image.open(img_stream)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Make predictions
def predict_result(input_batch):
   
    # return str(input_batch)
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_label = "xxxx"
    predicted_label = classes[predicted_idx.item()]

    # Print the predicted label
    print('Predicted Label:', predicted_label)
    return predicted_label



def preprocess_img_old(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result_old(img):
    # pred = model.predict(img)
    # result = np.argmax(pred[0], axis=-1)
    return "dupa"
