import pandas as pd
from torch.utils.data import DataLoader
from utils.dataset import ImageDataset

def load_train_test_valid_df(csv_filename="data/bird_df.csv"):
    """
    Load train, test, and valid data from csv file into Pandas dataframes.

    Parameters
    ----------
    csv_filename : str
        Filename/Path to the dataset csv file.

    Returns
    -------
    train_df : Dataframe
        Dataframe holding training data
    test_df : Dataframe
        Dataframe holding testing data
    valid_df : Dataframe
        Dataframe hold validation data

    Example
    -------
    csv_filename = "data/bird_df.csv"
    train_df, test_df, valid_df = load_train_test_valid_df(csv_filename)
    """
    df = pd.read_csv(csv_filename)
    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']
    valid_df = df[df['dataset'] == 'valid']
    return train_df, test_df, valid_df


def get_train_test_valid_dataloaders(train_df, test_df, valid_df):
    """
    Create train, test, and valid PyTorch dataloaders from Pandas dataframes.
    Each dataloader wraps around a dataset.ImageDataset to iterate and load
    samples of the dataset.

    Parameters
    ----------
    train_df : Dataframe
        Dataframe holding training data
    test_df : Dataframe
        Dataframe holding testing data
    valid_df : Dataframe
        Dataframe hold validation data

    Returns
    -------
    train_dl : Dataloader
    test_dl : Dataloader
    valid_dl : Dataloader

    Example
    -------
    csv_filename = "data/bird_df.csv"
    train_df, test_df, valid_df = load_train_test_valid_df(csv_filename)
    train_dl, test_dl, valid_dl = get_train_test_valid_dataloaders(
        train_df, test_df, valid_df
    )
    """
    train_dataset = ImageDataset(train_df, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = ImageDataset(test_df, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    valid_dataset = ImageDataset(valid_df, train=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader, valid_dataloader
