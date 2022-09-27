from models.xresnet1d import xresnet1d50
from models.seresnet2d import se_resnet34
import torch.nn.functional as F
import torch
import torch.nn as nn

class ensemble_model(nn.Module):
    def __init__(self, no_classes=24, gate=False, w_time=None, w_spec=None,device=None):
        super(ensemble_model,self).__init__()
        # gating encoding
        self.gate = gate 

        # Time series module 
        self.time_backbone = xresnet1d50(widen=1.0)
        time_list_of_modules = list(self.time_backbone.children())
        self.time_features = nn.Sequential(*time_list_of_modules[:-1], time_list_of_modules[-1][0])
        time_num_ftrs = self.time_backbone[-1][-1].in_features
        self.time_backbone[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)

        if w_time is not None:
            time_state_dict = torch.load(w_time,map_location=device)
            self.time_features.load_state_dict(time_state_dict,strict=False)

        self.spec_backbone = se_resnet34()
        self.spec_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        spec_list_of_modules = list(self.spec_backbone.children())
        self.spec_features = nn.Sequential(*spec_list_of_modules[:-1])
        spec_num_ftrs = self.spec_backbone.fc.in_features

        if w_spec is not None:
            spec_state_dict = torch.load(w_spec,map_location=device)
            self.spec_features.load_state_dict(spec_state_dict,strict=False)

        if self.gate:
            num_ftrs = time_num_ftrs + spec_num_ftrs
            self.gate_fc = nn.Linear(num_ftrs,2)
            self.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                nn.Linear(in_features=num_ftrs//2,out_features=no_classes)
            )
        else:
            num_ftrs = time_num_ftrs + spec_num_ftrs
            self.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                nn.Linear(in_features=num_ftrs//2,out_features=no_classes)
            )
            

    def forward(self, x_sig, x_spec):
        h_time = self.time_features(x_sig)
        h_time = h_time.squeeze()

        h_spec = self.spec_features(x_spec)
        h_spec = h_spec.squeeze()

        if self.gate:
            h_gate = F.softmax(self.gate_fc(torch.cat((h_time,h_spec),dim=1)),dim=1)
            h_encode = torch.cat([h_time*h_gate[:,0:1],h_spec*h_gate[:,1:2]],dim=1)
            x = self.fc(h_encode)
            return x 
        else:
            h_comb = torch.cat((h_time,h_spec),1)
            x = self.fc(h_comb)
            return x



class ensemble_model_3head(nn.Module):
    def __init__(self, no_classes=24,w_time=None, w_spec=None,device=None):
        super(ensemble_model_3head,self).__init__()

        # Time series module 
        self.time_backbone = xresnet1d50(widen=1.0)
        time_list_of_modules = list(self.time_backbone.children())
        self.time_features = nn.Sequential(*time_list_of_modules[:-1], time_list_of_modules[-1][0])
        time_num_ftrs = self.time_backbone[-1][-1].in_features
        self.time_backbone[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)

        if w_time is not None:
            time_state_dict = torch.load(w_time,map_location=device)
            self.time_features.load_state_dict(time_state_dict,strict=False)

        self.spec_backbone = se_resnet34()
        self.spec_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        spec_list_of_modules = list(self.spec_backbone.children())
        self.spec_features = nn.Sequential(*spec_list_of_modules[:-1])
        spec_num_ftrs = self.spec_backbone.fc.in_features

        if w_spec is not None:
            spec_state_dict = torch.load(w_spec,map_location=device)
            self.spec_features.load_state_dict(spec_state_dict,strict=False)

        
        num_ftrs = time_num_ftrs + spec_num_ftrs
        self.gate_fc = nn.Linear(num_ftrs,2)

        self.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
            nn.Linear(in_features=num_ftrs//2,out_features=no_classes)
            )
        self.fc_time = nn.Sequential(
            nn.Linear(in_features=time_num_ftrs,out_features=time_num_ftrs//2),
            nn.Linear(in_features=time_num_ftrs//2,out_features=no_classes)
            )
        self.fc_spec = nn.Sequential(
            nn.Linear(in_features=spec_num_ftrs,out_features=spec_num_ftrs//2),
            nn.Linear(in_features=spec_num_ftrs//2,out_features=no_classes)
            )

        # for p in self.fc.parameters():
        #     p.requires_grad = False
        # for p in self.gate_fc.parameters():
        #     p.requires_grad = False


    def forward(self, x_sig, x_spec):
        h_time = self.time_features(x_sig)
        h_time = h_time.squeeze()

        h_spec = self.spec_features(x_spec)
        h_spec = h_spec.squeeze()

        h_gate = F.softmax(self.gate_fc(torch.cat((h_time,h_spec),dim=1)),dim=1)
        h_encode = torch.cat([h_time*h_gate[:,0:1],h_spec*h_gate[:,1:2]],dim=1)
        y = self.fc(h_encode)
        y_time = self.fc_time(h_time)
        y_spec = self.fc_spec(h_spec)
        
        return y, y_time, y_spec, h_gate

    def freeze_backbone(self):
        for p in self.spec_features.parameters():
            p.requires_grad = False
        for p in self.time_features.parameters():
            p.requires_grad = False
    
    def freeze_gate(self):
        for p in self.gate_fc.parameters():
            p.requires_grad = False
