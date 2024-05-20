import torch.nn as nn
import torch

class CustomRMSE(nn.Module):
    def __init__(self):
        super(CustomRMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inA, inB):
        #assert predictions.keys() == labels.keys(), "Keys of predictions and labels must match"
        #loss = torch.sqrt(self.mse(inA, inB))
        return self.mse(inA, inB)