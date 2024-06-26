import os
from timeit import default_timer

import torch
import torch.nn as nn

from models.efficient_net import EfficientNet
from models.model1 import CNN
from models.pretrained_efficient_net import build_pretrained_efficient_net_model
from utils.data_utils import get_train_test_valid_dataloaders, load_train_test_valid_df
from utils.model_utils import save_model
from utils.test_utils import test_model
from utils.train_utils import plot_learning_curve, train

# model_types = ["efficient_net", "pretrained_efficient_net", "CNN"]

model_config = {
    "model_type": "pretrained_efficient_net",  # ["efficient_net", ...]
    "num_classes": 525,  # number of bird classes
    "lr": 0.001,  # learning rate
}

# torch.manual_seed(1)

if __name__ == "__main__":
    # Load train, test, and valid dataframes
    train_df, test_df, valid_df = load_train_test_valid_df(
        csv_filename="data/bird_df.csv"
    )

    # Create train, test, and valid dataloaders
    train_dl, test_dl, valid_dl = get_train_test_valid_dataloaders(
        train_df, test_df, valid_df
    )

    # Use GPU if available otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"model_type: {model_config['model_type']}")

    # Create model and send to device
    if model_config["model_type"] == "efficient_net":
        model = EfficientNet(
            version="b0",
            num_classes=model_config["num_classes"],
        ).to(device)
    elif model_config["model_type"] == "pretrained_efficient_net":
        model = build_pretrained_efficient_net_model()
        model.to(device)
    else:
        model = CNN(model_config["num_classes"]).to(device)

    # Print model architecture
    print("---------------------------------------------\n")
    print(model)
    print("\n---------------------------------------------")

    # Create loss_fn and optimzer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["lr"])

    # Specify number of epochs
    num_epochs = 20

    # Time training
    t1 = default_timer()
    model, hist = train(model, loss_fn, optimizer, num_epochs, train_dl, valid_dl)
    t2 = default_timer()

    print("-" * 20)
    print("")
    print(f"TIME: {t2-t1}")
    print("")
    print("-" * 20)

    # Plot training data
    plot_learning_curve(
        hist,
        save_image=True,
        filename="models/training_images/pretrained_efficient_net01.jpg",
    )

    # Make sure directory exists
    if not os.path.exists("models/saved_models"):
        os.mkdir("models/saved_models")

    # Save model
    # path = 'models/saved_models/efficient_net01.ph'
    path = "models/saved_models/pretrained_efficient_net01.ph"
    save_model(model, path)

    # Test the model
    test_model(test_dl, model, loss_fn)
