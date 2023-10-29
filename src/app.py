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

IMAGE_DIR = 'src/static/img'
IMAGE_PATH = Path(IMAGE_DIR)

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('main'))

@app.route('/main/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files.get('image')
        if image is None or image.filename == '':
            return jsonify({'error': 'no file'})

        filename = secure_filename(image.filename)
        filepath = IMAGE_PATH / filename

        #image.save(str(filepath))

    return jsonify({'filepath': str(filepath)})
