import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

def save_checkpoint(model, optimizer, filename):
    """
    Save a checkpoint of the model and optimizer.

    Parameters
    ----------
    model : PyTorch model
        Model to be saved
    optimizer : PyTorch optimizer
        Optimizer to be saved
    filename : str
        Filename/Path to write checkpoint data

    Example
    -------
    for epoch in range(num_epochs):
        ...

        # Save checkpoint
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, checkpoint_filename)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    """
    Load a model and optimizer from a saved checkpoint.

    Parameters
    ----------
    model : PyTorch model
        Model to load checkpoint data into
    optimizer : PyTorch optimizer
        Optimizer to load checkpoint data into
    filename : str
        Filename/Path to load checkpoint data

    Returns
    -------
    model
        A PyTorch model loaded from a saved checkpoint state_dict.
    optimizer
        A PyTorch optimizer loaded from a saved checkpoint state_dict.

    Example
    -------
    checkpoint_filename = "checkpoint_epoch_10.pt"
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model, optimizer = load_checkpoint(model, optimizer, checkpoint_filename)
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer



# TODO: clean up train func
def train(model, loss_fn, optimizer, num_epochs, train_dl, valid_dl):
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

    return model, (loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid)


def plot_learning_curve(train_history, save_image=False, filename=None):
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

    if save_image:
        plt.savefig(filename)

    plt.show()

