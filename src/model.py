import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os

from dataset import ImageDataset

from timeit import default_timer

df = pd.read_csv("bird_df.csv")
test_df = df[df['dataset'] == 'test']
train_df = df[df['dataset'] == 'train']
valid_df = df[df['dataset'] == 'valid']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-5,5)),
    transforms.ToTensor(),
])

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        ## Convolutional Layers
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        #self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Fully connected/linear layers
        #self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        #self.relu4 = nn.ReLU()
        #self.fc2 = nn.Linear(1024, num_classes)

        # --------------------------------

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        #self.fc1 = nn.Linear(64 * 56 * 56, 512)
        #self.fc1 = nn.Linear(64 * 28 * 28, 512)
        #self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        #self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(32 * 56 * 56, 1024)
        self.relu4 = nn.ReLU()
        #self.fc2 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #print(f'1) size: {x.shape}')
        x = self.pool1(self.relu1(self.conv1(x)))
        #print(f'2) size: {x.shape}')
        x = self.pool2(self.relu2(self.conv2(x)))
        #print(f'3) size: {x.shape}')
        #x = self.pool3(self.relu3(self.conv3(x)))
        #print(f'4) size: {x.shape}')

        x = torch.flatten(x, 1)
        #print("SIZE: ", x.shape)

        #x = x.view(-1, 64 * 28 * 28)
        x = self.fc1(x)
        #print(f'5) size: {x.shape}')
        #x = x.view(64*56*56, -1)
        #x = self.dropout1(x)
        x = self.relu4(self.fc1(x))
        #print(f'6) size: {x.shape}')
        x = self.fc2(x)
        #print(f'7) size: {x.shape}')

        return x

# Number of bird labels
num_classes = 525

# Instantiate the CNN model
model = CNN(num_classes)

# Print the model architecture
print(model)

# Create datasets and dataloaders
train_dataset = ImageDataset(train_df, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ImageDataset(test_df, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

valid_dataset = ImageDataset(valid_df, transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
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

        i = 0

        for x_batch, y_batch in train_dl:

            if i%100 == 0:
                print(f"batch: {i}")
            i += 1

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            #print(f'batch_size: {len(x_batch)}')
            #print(f'batch_size: {len(y_batch)}')
            ##x_batch = x_batch.to(device)
            ##y_batch = y_batch.to(device)
            #print(f'typeof(x_batch): {type(x_batch)}')
            pred = model(x_batch)
            #print(f'pred size: {len(pred)}')
            #print(f'typeof(pred): {type(pred)}')
            #print(f'typeof(y_batch): {type(y_batch)}')
            #print(f'pred:\n {pred}')
            #print(f'y_batch:\n {y_batch}')
            #print(f'shape pred:\n {pred.shape}')
            #print(f'shape y_batch:\n {y_batch.shape}')

            #print(f'size of pred: {pred.shape}')
            #print(f'pred[0]: {pred[0]}')
            #print(f'size of y_batch: {y_batch.shape}')
            #print(f'y_batch[0]: {y_batch[0]}')

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

#torch.manual_seed(1)
num_epochs = 20

t1 = default_timer()
hist = train(model, num_epochs, train_dataloader, valid_dataloader)
t2 = default_timer()

print("-"*20)
print("")
print(f"TIME: {t2-t1}")
print("")
print("-"*20)

x_arr = np.arange(len(hist[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.savefig('images/train04.png')
plt.show()


#torch.cuda.synchronize()
#model_cpu = model.cpu()
#pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)
#is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
#print(f'Test accuracy: {is_correct.mean():.4f}')
#
#
#
#
#fig = plt.figure(figsize=(12, 4))
#for i in range(12):
#    ax = fig.add_subplot(2, 6, i+1)
#    ax.set_xticks([]); ax.set_yticks([])
#    img = mnist_test_dataset[i][0][0, :, :]
#    pred = model(img.unsqueeze(0).unsqueeze(1))
#    y_pred = torch.argmax(pred)
#    ax.imshow(img, cmap='gray_r')
#    ax.text(0.9, 0.1, y_pred.item(),
#            size=15, color='blue',
#            horizontalalignment='center',
#            verticalalignment='center',
#            transform=ax.transAxes)
#
#
#plt.savefig('figures/14_14.png')
#plt.show()


if not os.path.exists('models'):
    os.mkdir('models')

path = 'models/model04.ph'
torch.save(model, path)


