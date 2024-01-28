import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        imgpath = self.df.iloc[idx]['filepath']
        #label = self.df.iloc[idx]['label']
        label = self.df.iloc[idx]['class_id']

        # Load image
        image = Image.open(imgpath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

df = pd.read_csv("bird_df.csv")
test_df = df[df['dataset'] == 'test']
train_df = df[df['dataset'] == 'train']
valid_df = df[df['dataset'] == 'valid']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = ImageDataset(train_df, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ImageDataset(test_df, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

valid_dataset = ImageDataset(valid_df, transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

for i, batch in enumerate(train_dataloader):
    print(f'batch {i}: {batch}')
    print("---------------------------------------------")
    break
