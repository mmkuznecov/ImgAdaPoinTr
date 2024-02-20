import torch.nn as nn
import torchvision.models as models
import timm


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # This is an abstract class, so we don't initialize a specific model here.

    def forward(self, x):
        # The forward pass needs to be implemented by subclasses.
        raise NotImplementedError("Subclass must implement forward method.")


class ConvNext(ImageEncoder):
    def __init__(self):
        super().__init__()
        base = timm.create_model("convnext_small_384_in22ft1k", pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.base(x)
        x = x.reshape(x.size(0), 98, -1)
        return x


class ResNet18(ImageEncoder):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-3])

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), 256, -1)
        return x


class ResNet50(ImageEncoder):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), 8, -1)
        return x
