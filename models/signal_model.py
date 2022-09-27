from models.xresnet1d import xresnet1d50
import torch.nn as nn

class signal_model(nn.Module):
    def __init__(self,no_classes):
        super(signal_model,self).__init__()
        self.backbone = xresnet1d50(widen=1.0)
        list_of_modules = list(self.backbone.children())

        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
        self.num_ftrs = self.backbone[-1][-1].in_features
        self.backbone[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)

        self.fc = nn.Sequential(
                nn.Linear(in_features=self.num_ftrs,out_features=self.num_ftrs//2),
                nn.Linear(in_features=self.num_ftrs//2,out_features=no_classes)
            )

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.fc(h)
        return x

class signal_model_simclr(nn.Module):
    def __init__(self,no_classes):
        super(signal_model_simclr,self).__init__()
        self.backbone = xresnet1d50(widen=1.0)
        list_of_modules = list(self.backbone.children())

        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
        self.num_ftrs = self.backbone[-1][-1].in_features
        self.backbone[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)

        self.fc = nn.Sequential(
                nn.Linear(in_features=self.num_ftrs,out_features=self.num_ftrs//2),
                nn.Linear(in_features=self.num_ftrs//2,out_features=no_classes)
            )

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.fc(h)
        return h, x


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class signal_model_byol(nn.Module):
    def __init__(self,no_classes):
        super(signal_model_byol,self).__init__()
        self.backbone = xresnet1d50(widen=1.0)
        list_of_modules = list(self.backbone.children())

        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
        num_ftrs = self.backbone[-1][-1].in_features
        self.num_ftrs = num_ftrs
        self.backbone[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)

        self.projection = MLPHead(in_channels=num_ftrs,mlp_hidden_size=512,projection_size=128)
        self.predictor = MLPHead(in_channels=128,mlp_hidden_size=512,projection_size=128)

        self.fc = nn.Sequential(
                nn.Linear(in_features=self.num_ftrs,out_features=self.num_ftrs//2),
                nn.Linear(in_features=self.num_ftrs//2,out_features=no_classes)
            )

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        z = self.projection(h)
        t = self.predictor(z)

        x = self.fc(h)
        return h,z,t,x  # features, projector, predictor, output