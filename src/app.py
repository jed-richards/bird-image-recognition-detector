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

app = Flask(__name__)

IMAGE_DIR = 'src/static/images'
IMAGE_PATH = Path(IMAGE_DIR)

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

    image.save(str(filepath))

    print(str(filepath))

    prediction = {
        "species": "species_name",
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

    return render_template('index.html', prediction=prediction)

    #return jsonify({'filepath': str(filepath)})
