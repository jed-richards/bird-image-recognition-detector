from flask import (
    Flask,
    render_template,
    abort,
    request,
    redirect,
    url_for,
    jsonify
)

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'result': 1})
    #return 'Results of GET'
