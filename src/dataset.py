from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    # TODO: look into making faster (bottlenecking GPU)
    def __getitem__(self, idx):
        imgpath = self.df.iloc[idx]['filepath']
        label = self.df.iloc[idx]['class_id']

        # Load image
        image = Image.open(imgpath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
