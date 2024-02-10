import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNN  # my CNN class
from dataset import ImageDataset  # my ImageDataset class

# Function to load the model
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Load model
#MODEL_PATH = 'src/models/model03.ph'
#model = load_model(MODEL_PATH)

#torch.save(model.state_dict(), 'src/models/model03_sd.ph')

#exit()

MODEL_PATH = 'src/models/model03_sd.ph'
model = CNN(num_classes=525)
model.load_state_dict(torch.load(MODEL_PATH))


# Print model architecture
print(model)

# Read and load test dataset
df = pd.read_csv("data/bird_df.csv")
test_df = df[df['dataset'] == 'test']

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = ImageDataset(test_df, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
loss_fn = nn.CrossEntropyLoss()

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

test_loop(test_dataloader, model, loss_fn)
