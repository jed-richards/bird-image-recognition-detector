import torch
import torch.nn as nn
from math import ceil

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from timeit import default_timer
from dataset import ImageDataset  # my ImageDataset class

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class  CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1
        ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # Sigmoid Linear Unit (activation) SiLu <-> Swish

    def forward(self, x):
        return self.silu(self.batch_norm(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.squeeze_excitation = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Basically tells us how much we are prioritizing each channel.
        """
        return x * self.squeeze_excitation(x)

class InvertedResidualBlock(nn.Module):
    """
    Look at MobileNetV2 for more information
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,        # for squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)  # integer division

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1, ): # alpha for depth scaling and beta for width scaling
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi  # how much times layers should increase each stage
        width_factor = beta ** phi   # how channels increase
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)  # make sure divisible by 4 for reduction
            layers_repeat = ceil(repeats * depth_factor)

            for layer in range(layers_repeat):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1, # only downsize on first layer of each stage
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # if k=1 : pad=0, k=3 : pad=1, k=5 : pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1)) # flatten to send through classifier


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
    ).to(device)
    print(model)
    print(model(x).shape)  # (num_examples, num_classes)


#if __name__ == "__main__":
#    test()

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
    #model = CNN(num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_classes = 525
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
    ).to(device)

    # Print the model architecture
    print(model)

    # Loss and Optimzer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

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

    #if not os.path.exists('models'):
    #    os.mkdir('models')

    # Save model
    path = 'models/efficient_net01.ph'
    #torch.save(model, path)

    # Save only state_dict
    torch.save(model.state_dict(), path)

    # Test the model
    #test_model(test_dataloader, model, loss_fn)
