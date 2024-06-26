# B.I.R.D.
B.I.R.D. or Bird Image Recognition Detector is a machine learning model to classify the species of birds from images.

## Installation

### Dependencies
- Install [conda](https://docs.conda.io/en/latest/)

```
git clone https://github.com/jed-richards/bird-image-recognition-detector
cd bird-image-recognition-detector
conda env create -f environment.yml
conda activate bird
```

### Bird Image Dataset
- Download [Kaggle - Birds 525 Species Dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).
```
unzip PATH/TO/archive.zip -d PATH/TO/bird-image-recognition-detector/data
```

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
