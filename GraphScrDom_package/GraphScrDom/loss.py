import torch
import torch.nn as nn
import torch.nn.functional as F

class SCLLoss(nn.Module):
    def __init__(self, weight_mse=10, weight_scl=1):
        """
        Custom loss class for SCL, which combines MSE loss and BCEWithLogitsLoss.

        Args:
            weight_mse (float): Weight for the MSE loss component.
            weight_scl (float): Weight for the SCL loss components (BCEWithLogitsLoss).
        """
        super(SCLLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.scl_loss = nn.BCEWithLogitsLoss()
        self.weight_mse = weight_mse
        self.weight_scl = weight_scl

    def forward(self, features, recon_features, ret, ret_a, label_SCL):
        """
        Forward pass for computing the combined SCL loss.

        Args:
            features (torch.Tensor): Original input features.
            recon_features (torch.Tensor): Reconstructed features from the model.
            ret (torch.Tensor): Output from the SCL model for the original data.
            ret_a (torch.Tensor): Output from the SCL model for the augmented data.
            label_SCL (torch.Tensor): Labels for the SCL task (binary labels).

        Returns:
            torch.Tensor: The combined loss value.
        """
        # Compute MSE loss between original features and reconstructed features
        loss_recon = self.mse_loss(features, recon_features)

        # Compute SCL loss (BCEWithLogitsLoss) for both original and augmented outputs
        loss_sl_1 = self.scl_loss(ret, label_SCL)
        loss_sl_2 = self.scl_loss(ret_a, label_SCL)

        # Combine the MSE and SCL losses
        total_loss = self.weight_mse * loss_recon + self.weight_scl * (loss_sl_1 + loss_sl_2)

        return total_loss
    
    
class UnsupLoss(nn.Module):
    def __init__(self):
        super(UnsupLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, inds_unsup):
        _, target = torch.max(output, dim=1)
        
        loss = self.cross_entropy(output[inds_unsup], target[inds_unsup])

        return loss
    
    
class SupLoss(nn.Module):
    def __init__(self):
        super(SupLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, inds_sup, target_scr):
        loss = self.cross_entropy(output[inds_sup], target_scr[inds_sup])
        return loss
