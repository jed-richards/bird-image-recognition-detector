import torch


def save_model(model, filename):
    """
    Save the final model's state_dict to a file.

    Parameters
    ----------
    model : PyTorch model
        Model to be saved
    filename : str
        Filename/Path to save model
    """
    torch.save(model.state_dict(), filename)


def load_model(model, model_path):
    """
    Load a saved model from state_dict and put in evaluation mode.

    Parameters
    ----------
    model : PyTorch model
        A PyTorch model object to load the saved model's parameters into.
        This model must have the same network architecture as the saved
        model.
    model_path : str
        File path to saved model, e.g. "path/to/model.ph".

    Returns
    -------
    model
        A PyTorch model loaded from a saved state_dict.

    Example
    -------
    model_path = 'path/to/model.ph'
    model = CNN(num_classes=525)
    model = load_model(model, model_path)
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()  # put in evaluation mode
    return model
