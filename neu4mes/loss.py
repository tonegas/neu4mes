import torch.nn as nn
import torch

class CustomRMSE(nn.Module):
    def __init__(self):
        super(CustomRMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, labels):
        assert predictions.keys() == labels.keys(), "Keys of predictions and labels must match"

        losses = []
        for key in predictions.keys():
            pred = predictions[key]
            label = labels[key]
            loss = torch.sqrt(self.mse(pred, label))
            losses.append(loss)

        # Calculate the mean RMSE over all keys
        rmse_loss = torch.mean(torch.stack(losses))
        
        return rmse_loss