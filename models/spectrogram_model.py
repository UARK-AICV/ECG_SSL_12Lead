from models.seresnet2d import se_resnet34
import torch.nn as nn

class spectrogram_model(nn.Module):
    def __init__(self,no_classes):
        super(spectrogram_model,self).__init__()
        self.backbone = se_resnet34()
        self.backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        list_of_modules = list(self.backbone.children())
        self.features = nn.Sequential(*list_of_modules[:-1])
        num_ftrs = self.backbone.fc.in_features
    
        self.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                nn.Linear(in_features=num_ftrs//2,out_features=no_classes)
            )

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.fc(h)
        return x
