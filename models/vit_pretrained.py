import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16

# Model Selecting
def get_vit_model(model_type, num_classes):
    if model_type == 'vit_b_16':
        model = vit_b_16(weights="IMAGENET1K_V1")
    elif model_type == 'vit_b_32':
        model = vit_b_32(weights='IMAGENET1K_V1')
    elif model_type == 'vit_l_16':
        model = vit_l_16(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # modify the classifier head for CIFAR-100
    if isinstance(model.heads, nn.Sequential):
        in_features = model.heads[-1].in_features
    else:
        in_features = model.heads.in_features
    model.heads = nn.Linear(in_features, num_classes)
    return model