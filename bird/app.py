# TODO: clean this up
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

# Append "src" to the Python runtime path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# from model import CNN  # my CNN class
# from models.efficient_net import EfficientNet
from models.pretrained_efficient_net import build_pretrained_efficient_net_model

app = Flask(__name__)

IMAGE_DIR = "bird/static/images"
IMAGE_PATH = Path(IMAGE_DIR)

if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)


# Function to load the model
def load_model(model_path):
    # model = CNN(num_classes=525)
    # model = EfficientNet(version="b0", num_classes=525)
    model = build_pretrained_efficient_net_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    # Use GPU or CPU
    # model.to(device)

    return model


# Load model
# MODEL_PATH = 'src/models/model03_sd.ph'
MODEL_PATH = "models/saved_models/pretrained_efficient_net01.ph"
model = load_model(MODEL_PATH)


# Transform input into the form our model expects
def transform_image(image_file) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # We use multiple TorchVision transforms to ready the image
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_file)  # Open the image file
    img_tensor = transform(
        image
    )  # Transform PIL image to appropriately-shaped PyTorch tensor
    img_tensor.unsqueeze_(0)  # PyTorch models expect batched input; create a batch of 1
    return img_tensor


# Creates a dictionary of class_id to [bird_name, bird_image_path]
def get_labels_dict() -> Dict[str, List[str]]:
    df = pd.read_csv("data/bird_df.csv")
    test_df = df[df["dataset"] == "test"]
    labels_dict = (
        test_df.groupby("class_id")
        .apply(lambda x: [x["label"].iloc[0], x["filepath"].iloc[0]])
        .to_dict()
    )
    return labels_dict


labels_dict = get_labels_dict()


def get_top_three_predictions(input_tensor: torch.Tensor) -> List[Dict[str, any]]:
    with torch.no_grad():
        output = model(input_tensor)
    output = torch.nn.functional.softmax(output[0], dim=0)
    confidence, indices = torch.topk(
        output, k=3
    )  # Get top 3 indices and their confidence scores
    top_three_predictions = []
    for confidence_score, index in zip(confidence, indices):
        prediction_id = index.item()
        confidence_score = confidence_score.item()

        confidence_score = confidence_score * 100  # convert to percentage

        bird_label, bird_img_path = labels_dict[prediction_id]
        top_three_predictions.append(
            {
                "predicted_id": prediction_id,
                "bird_label": bird_label,
                "bird_img_path": bird_img_path,
                "confidence": confidence_score,
            }
        )
    return top_three_predictions


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("main"))


@app.route("/main/", methods=["GET"])
def main():
    return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def predict():
    image = request.files.get("image")
    if image is None or image.filename == "":
        return render_template("index.html", error="Not a valid image/image file name")

    filename = secure_filename(image.filename)
    filepath = IMAGE_PATH / filename

    image = Image.open(image)  # Open the image file
    image = image.resize((224, 224))  # Resize image

    image.save(str(filepath))
    print(str(filepath))

    img_tensor = transform_image(filepath)

    top_three_predictions = get_top_three_predictions(img_tensor)

    prediction1 = {
        "species": top_three_predictions[0]["bird_label"],
        "predicted_id": top_three_predictions[0]["predicted_id"],
        "accuracy": f'{top_three_predictions[0]["confidence"]:.2f}',
        "predicted_image": top_three_predictions[0]["bird_img_path"],
    }

    prediction2 = {
        "species": top_three_predictions[1]["bird_label"],
        "predicted_id": top_three_predictions[1]["predicted_id"],
        "accuracy": f'{top_three_predictions[1]["confidence"]:.2f}',
        "predicted_image": top_three_predictions[1]["bird_img_path"],
    }

    prediction3 = {
        "species": top_three_predictions[2]["bird_label"],
        "predicted_id": top_three_predictions[2]["predicted_id"],
        "accuracy": f'{top_three_predictions[2]["confidence"]:.2f}',
        "predicted_image": top_three_predictions[2]["bird_img_path"],
    }

    prediction = {
        "user_image": str(filepath)[5:],  # remove 'bird/'
        "prediction1": prediction1,
        "prediction2": prediction2,
        "prediction3": prediction3,
    }

    print(jsonify({"filepath": str(filepath)}))
    print(jsonify({"filepath[5:]": str(filepath)[5:]}))
    print(jsonify(prediction))
    # return jsonify(prediction)
    return render_template("index.html", prediction=prediction)


@app.route("/image/<path:predicted_image_path>")
def serve_image(predicted_image_path):
    # added the "../" because it sends the file and attaches the .../src/data/test/...
    # so now it will have .../src/../data/test/...
    predicted_image_path = "../" + predicted_image_path
    # print(predicted_image_path)
    # predicted_image_path = os.path.abspath(predicted_image_path)
    print(predicted_image_path)
    # print(predicted_image_path)
    # image_dir = Path(predicted_image_path).parent
    # print(image_dir)

    # return send_from_directory(image_dir, predicted_image_path)
    return send_file(predicted_image_path, mimetype="image/jpeg")
