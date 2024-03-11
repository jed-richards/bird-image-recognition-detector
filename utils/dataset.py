from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Default training transform
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
    ]
)

# Default testing and validation transform
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class ImageDataset(Dataset):
    """
    This is a custom dataset for the bird image dataset. This class
    implements the abstract class PyTorch Dataset.

    Attributes
    ----------
    df : Pandas DataFrame
        Pandas DataFrame representing the dataset.
    transform : torchvision.tranforms (default value of None)
        The transformation to be applied to images when loaded.
        If None is provided a default transform will be applied.
    train : bool (default value of True)
        Whether the dataset is for training or not.
    """

    def __init__(self, df: DataFrame, transform=None, train=True):
        self.df = df
        self.train = train
        if transform:
            self.transform = transform
        else:
            self.transform = train_transform if self.train else test_transform

    def __len__(self):
        return self.df.shape[0]

    # TODO: look into making faster (bottlenecking GPU)
    def __getitem__(self, idx):
        imgpath = self.df.iloc[idx]["filepath"]
        label = self.df.iloc[idx]["class_id"]

        # Load image
        image = Image.open(imgpath).convert("RGB")

        # Apply image transformation
        image = self.transform(image)

        return image, label
