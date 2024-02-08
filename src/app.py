from flask import (
    Flask,
    render_template,
    abort,
    request,
    redirect,
    url_for,
    jsonify
)
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from torchvision import transforms
import torch
from PIL import Image
from model import CNN  # my CNN class

#app = Flask(__name__)
app = Flask("src")

IMAGE_DIR = 'src/static/images'
IMAGE_PATH = Path(IMAGE_DIR)

if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

# Load model
#MODEL_PATH = "models/model01.ph"
#model = torch.load(MODEL_PATH)
#model.eval()

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

# Get a prediction
#def get_prediction(input_tensor):
#    outputs = model(input_tensor)
#    print(f'outputs: {outputs}')
#    _, y_hat = outputs.max(1)              # Extract the most likely class
#    prediction_id = y_hat.item()           # Extract the int value from the PyTorch tensor
#    print(f'pred_id: {prediction_id}')
#    return prediction_id


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

    #img_tensor = transform_image(filepath)
    #prediction_id = get_prediction(img_tensor)
    #print(f'predicted_id = {prediction_id}')

    prediction = {
        "species": "species_name",
    #    "predicted_id": prediction_id,
        "accuracy": "percentage",
        "user_image": str(filepath)[4:], # remove 'src/'
        "predicted_image": str(filepath)[4:], # this will need to be changed to the predicted image path
    }

    # Used for image in powerpoint
    #predicted_image_path = filepath.with_stem(filepath.stem + "-PREDICTED")
    #prediction = {
    #    "species": filepath.stem,
    #    "accuracy": 94.2,
    #    "user_image": str(filepath)[4:], # remove 'src/'
    #    "predicted_image": str(predicted_image_path)[4:], # this will need to be changed to the predicted image path
    #}

    print(jsonify({'filepath': str(filepath)}))

    return render_template('index.html', prediction=prediction)

    #return jsonify({'filepath': str(filepath)})
