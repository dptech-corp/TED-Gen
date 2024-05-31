import torch.nn as nn
import torchvision.models as models

class StructNet(nn.Module):
    def __init__(self,output_dim):
        super(StructNet, self).__init__()     
        self.backbone = models.resnet50(weights=None)
        self.shift = nn.Sequential(
            nn.Linear(1000,512), 
            nn.ReLU(),
            nn.Linear(512,output_dim),
        )
        
    def forward(self,x):
        x = self.backbone(x)
        return self.shift(x)