import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from timeit import default_timer
from dataset import ImageDataset  # my ImageDataset class
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        ## Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected/linear layers
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Conv Layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Fully connected layers
        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x


def train(model, num_epochs, train_dl, valid_dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"device: {device}")
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        ti = default_timer()
        print(f"epoch: {epoch}")
        model.train()

        for i, (x_batch, y_batch) in enumerate(train_dl):
            if i%100 == 0:
                print(f"batch: {i}")

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        tf = default_timer()
        print(f'train time: {tf-ti}')

        tii = default_timer()

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')

        tf = default_timer()
        print(f'valid time: {tf-tii}')

        print(f'total time: {tf-ti}')

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


def plot_learning_curve(train_history):
    x_arr = np.arange(len(train_history[0])) + 1
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, train_history[0], '-o', label='Train loss')
    ax.plot(x_arr, train_history[1], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, train_history[2], '-o', label='Train acc.')
    ax.plot(x_arr, train_history[3], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)

    #plt.savefig('images/train04.png')
    plt.show()

def test_model(dataloader, model, loss_fn):
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

############################### Train Model ###################################

if __name__ == "__main__":
    # Read and load dataset
    df = pd.read_csv("data/bird_df.csv")
    test_df = df[df['dataset'] == 'test']
    train_df = df[df['dataset'] == 'train']
    valid_df = df[df['dataset'] == 'valid']

    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-5,5)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    train_dataset = ImageDataset(train_df, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = ImageDataset(valid_df, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    test_dataset = ImageDataset(test_df, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Number of bird labels
    num_classes = 525

    # Instantiate the CNN model
    model = CNN(num_classes)

    # Print the model architecture
    print(model)

    # Loss and Optimzer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #torch.manual_seed(1)
    num_epochs = 20

    # Time training
    t1 = default_timer()
    hist = train(model, num_epochs, train_dataloader, valid_dataloader)
    t2 = default_timer()

    print("-"*20)
    print("")
    print(f"TIME: {t2-t1}")
    print("")
    print("-"*20)

    # Plot training data
    plot_learning_curve(hist)

    if not os.path.exists('models'):
        os.mkdir('models')

    # Save model
    #path = 'models/model04.ph'
    #torch.save(model, path)

    # Save only state_dict
    #torch.save(model.state_dict(), 'save/to/path/model.pth')

    # Test the model
    #test_model(test_dataloader, model, loss_fn)
