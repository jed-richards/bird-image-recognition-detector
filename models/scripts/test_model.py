#TODO:
# -[ ] clean this up and make it better
import torch
import torch.nn as nn
from models.efficient_net import EfficientNet
from utils.model_utils import load_model
from utils.data_utils import (
    get_train_test_valid_dataloaders, load_train_test_valid_df
)
from utils.test_utils import test_model

MODEL_PATH = 'models/saved_models/efficient_net01.ph'
model = EfficientNet(version="b0", num_classes=525)
model = load_model(model, MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

# Print model architecture
print(model)

# Load train, test, and valid dataframes
train_df, test_df, valid_df = load_train_test_valid_df(
    csv_filename="data/bird_df.csv"
)

# Create train, test, and valid dataloaders
train_dl, test_dl, valid_dl = get_train_test_valid_dataloaders(
    train_df, test_df, valid_df
)

loss_fn = nn.CrossEntropyLoss()

test_model(test_dl, model, loss_fn)
