import torch.nn as nn
import torchvision.models as models


def build_pretrained_efficient_net_model(num_classes=525):
    model = models.efficientnet_b0(pretrained=True)

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
