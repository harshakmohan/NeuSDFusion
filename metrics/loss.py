import torch.nn as nn
import torch
import torch.autograd as autograd

class GeometryLoss(nn.Module):
    def __init__(self, sdf_weight=1.0, normal_weight=1.0, eikonal_weight=1.0):
        super(GeometryLoss, self).__init__()
        self.sdf_loss = SDFLoss()
        self.normal_loss = NormalLoss()
        self.eikonal_loss = EikonalLoss()

    def forward(self, predicted_sdf, pred_sdf_grad, gt_points, gt_sdf, gt_normals):

        # Split on-surface and off-surface points
        # For each point, if magnitude of gt_normal is zero, then it is an off-surface point
        on_surface_mask = gt_normals.norm(p=2, dim=1) > 0
        off_surface_mask = ~on_surface_mask

        # Use the mask to filter values for on-surface and off-surface points
        on_surface_points = gt_points[on_surface_mask]
        on_surface_predicted_sdf = predicted_sdf[on_surface_mask]
        on_surface_gt_sdf = gt_sdf[on_surface_mask]
        on_surface_pred_sdf_grad = pred_sdf_grad[on_surface_mask]
        on_surface_gt_normals = gt_normals[on_surface_mask]

        # Similarly, filter values for off-surface points using the off_surface_mask
        off_surface_points = gt_points[off_surface_mask]
        off_surface_predicted_sdf = predicted_sdf[off_surface_mask]
        off_surface_gt_sdf = gt_sdf[off_surface_mask]
        off_surface_pred_sdf_grad = pred_sdf_grad[off_surface_mask]

        sdf_loss_value = self.sdf_loss(on_surface_predicted_sdf, off_surface_predicted_sdf, off_surface_gt_sdf)
        normal_loss_value = self.normal_loss(on_surface_pred_sdf_grad, on_surface_gt_normals)
        eikonal_loss_value = self.eikonal_loss(on_surface_pred_sdf_grad)

        total_loss = sdf_loss_value + normal_loss_value + eikonal_loss_value
        return total_loss
    
class SDFLoss(nn.Module):
    def __init__(self, weight_1=1.0, weight_2=1.0):
        super(SDFLoss, self).__init__()
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def forward(self, predicted_on_surface_sdf, predicted_off_surface_sdf, gt_off_surface_sdf):
        '''
        Should take in predicted on-surface sdf values, predicted off-surface sdf values, and ground truth off-surface sdf values 
        '''
        # For on-surface points, compute the L1 norm of the predicted SDF values
        on_surface_loss = torch.norm(predicted_on_surface_sdf, p=1)
        # For off-surface points, compute the L1 norm of the difference between predicted and GT SDF values
        off_surface_loss = torch.norm(predicted_off_surface_sdf - gt_off_surface_sdf, p=1)
        # Weight the losses and sum them
        total_loss = self.weight_1 * on_surface_loss + self.weight_2 * off_surface_loss
        return total_loss


class EikonalLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(EikonalLoss, self).__init__()
        self.weight = weight

    def forward(self, pred_sdf_grad):
        # Calculate the magnitude of each gradient vector
        grad_magnitudes = torch.norm(pred_sdf_grad, p=2, dim=1)  # L2 norm across vector components
        
        # Compute the Eikonal loss: | ||grad SDF|| - 1 |
        loss = torch.abs(grad_magnitudes - 1)
        
        # Average loss over all samples and apply weight
        return self.weight * torch.mean(loss)


class NormalLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(NormalLoss, self).__init__()
        self.lambda_ = weight

    def forward(self, pred_sdf_grad, gt_on_surface_normals):
        # Compute the L1 norm of the difference between the predicted gradient and the ground truth normals
        difference = pred_sdf_grad - gt_on_surface_normals
        loss = torch.norm(difference, p=1, dim=1)  # L1 norm along the dimension of the vector components
        loss = self.lambda_ * loss.mean()  # Average over all points and scale by lambda
        return loss