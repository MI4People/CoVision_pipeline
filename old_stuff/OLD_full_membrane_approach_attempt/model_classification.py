from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Load pre-trained model
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify last fc layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=1, bias=True)

    def forward(self, x):
        y = F.sigmoid(self.model(x))
        return y
