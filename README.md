# B.I.R.D.
B.I.R.D. or Bird Image Recognition Detector is a machine learning model to classify the species of birds from images.

## Installation

```
git clone https://github.com/jed-richards/bird-image-recognition-detector
cd bird-image-recognition-detector
python3 -m venv env
pip install -r requirements.txt
```

## Running app 

``` 
make run-app
```
This will run the Flask server on port 5000. You can access the serve at `http://localhost:5000/`.

## Testing

```
coverage run -m pytests
coverage report -m
```

or 

```
make tests
```
