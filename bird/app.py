#TODO: clean this up
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_file,
)
from werkzeug.utils import secure_filename
from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd

import sys
import os

from typing import Dict, List

# Append "src" to the Python runtime path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

#from model import CNN  # my CNN class
from models.efficient_net import EfficientNet

app = Flask(__name__)

IMAGE_DIR = 'src/static/images'
IMAGE_PATH = Path(IMAGE_DIR)

if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

# Function to load the model
def load_model(model_path):
    #model = CNN(num_classes=525)
    model = EfficientNet(version="b0", num_classes=525)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load model
#MODEL_PATH = 'src/models/model03_sd.ph'
MODEL_PATH = 'src/models/efficient_net01.ph'
model = load_model(MODEL_PATH)

# Transform input into the form our model expects
def transform_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # We use multiple TorchVision transforms to ready the image
        transforms.ToTensor(),
    ])
    image = Image.open(image_file)       # Open the image file
    img_tensor = transform(image)        # Transform PIL image to appropriately-shaped PyTorch tensor
    img_tensor.unsqueeze_(0)             # PyTorch models expect batched input; create a batch of 1
    return img_tensor

# Creates a dictionary of class_id to [bird_name, bird_image_path]
def get_labels_dict() -> Dict[str, List[str]]:
    df = pd.read_csv("data/bird_df.csv")
    test_df = df[df['dataset'] == 'test']
    labels_dict = test_df.groupby('class_id').apply(
        lambda x: [x['label'].iloc[0], x['filepath'].iloc[0]]
    ).to_dict()
    return labels_dict

labels_dict = get_labels_dict()

# Get a prediction
def get_prediction(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    output = torch.nn.functional.softmax(output[0], dim=0)
    confidence, index = torch.max(output, 0)
    prediction_id = index.item()
    confidence = confidence.item()
    bird_label, bird_img_path = labels_dict[prediction_id]
    return prediction_id, bird_label, bird_img_path, confidence

    #outputs = model(input_tensor)
    ##print(f'outputs: {outputs}')
    #_, y_hat = outputs.max(1)              # Extract the most likely class
    #prediction_id = y_hat.item()           # Extract the int value from the PyTorch tensor
    #print(f'pred_id: {prediction_id}')
    #bird_label, bird_img_path = labels_dict[prediction_id]
    #return prediction_id, bird_label, bird_img_path


@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('main'))

@app.route('/main/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    image = request.files.get('image')
    if image is None or image.filename == '':
        return render_template('index.html', error="Not a valid image/image file name")

    filename = secure_filename(image.filename)
    filepath = IMAGE_PATH / filename

    image = Image.open(image)        # Open the image file
    image = image.resize((224, 224)) # Resize image

    image.save(str(filepath))
    print(str(filepath))

    img_tensor = transform_image(filepath)
    prediction_id, bird_label, bird_img_path, confidence = get_prediction(img_tensor)
    print(f'predicted_id = {prediction_id}')
    print(f'bird_label = {bird_label}')
    print(f'bird_img_path = {bird_img_path}')
    print(f'confidence = {confidence}')

    prediction = {
        "species": bird_label,
        "predicted_id": prediction_id,
        "accuracy": confidence,
        "user_image": str(filepath)[4:], # remove 'src/'
        "predicted_image": bird_img_path,
    }

    print(jsonify({'filepath': str(filepath)}))
    return jsonify(prediction)
    #return render_template('index.html', prediction=prediction)

@app.route('/image/<path:predicted_image_path>')
def serve_image(predicted_image_path):
    # added the "../" because it sends the file and attaches the .../src/data/test/...
    # so now it will have .../src/../data/test/...
    predicted_image_path = "../" + predicted_image_path
    #print(predicted_image_path)
    #predicted_image_path = os.path.abspath(predicted_image_path)
    print(predicted_image_path)
    #print(predicted_image_path)
    #image_dir = Path(predicted_image_path).parent
    #print(image_dir)

    #return send_from_directory(image_dir, predicted_image_path)
    return send_file(predicted_image_path, mimetype='image/jpeg')
