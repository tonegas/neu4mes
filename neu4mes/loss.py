import torch.nn as nn
import torch
from neu4mes.utilis import check

available_losses = ['mse', 'rmse', 'mae']

'''
class CustomRMSE(nn.Module):
    def __init__(self):
        super(CustomRMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inA, inB):
        #assert predictions.keys() == labels.keys(), "Keys of predictions and labels must match"
        #loss = torch.sqrt(self.mse(inA, inB))
        return self.mse(inA, inB)
'''
    
class CustomLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super(CustomLoss, self).__init__()
        check(loss_type in available_losses, TypeError, f'The \"{loss_type}\" loss is not available. Possible losses are: {available_losses}.')
        self.loss_type = loss_type
        self.loss = nn.MSELoss()
        if self.loss_type == 'mae':
            self.loss = nn.L1Loss()
        
    def forward(self, inA, inB):
        res = self.loss(inA,inB)
        if self.loss_type == 'rmse':
            res = torch.sqrt(res)
        return res