# B.I.R.D.
B.I.R.D. or Bird Image Recognition Detector is a machine learning model to classify the species of birds from images.

## Installation

```
git clone https://github.com/jed-richards/bird-image-recognition-detector
cd bird-image-recognition-detector
python3 -m venv env
pip install -r requirements.txt
```

### Bird Image Dataset
Download at [kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

## Running app 

``` 
make run-app
```
This will run the Flask server on port 5000. You can access the serve at `http://localhost:5000/`.

## Testing

```
make tests
```

or 

```
make test-coverage
```
