import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBinaryClassificationLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        l1_regularization_weight: float = 0.01,
        use_l1_reg: bool = True,
        class_weight: float = None,
        focal_loss_gamma: float = 2.0,
        use_focal_loss: bool = False,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.l1_regularization_weight = l1_regularization_weight
        self.use_l1_reg = use_l1_reg
        self.class_weight = class_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.use_focal_loss = use_focal_loss

    def forward(
        self,
        true_labels: torch.Tensor,
        predicted_logits: torch.Tensor,
        predicted_importance: torch.Tensor,
        original_importance: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_focal_loss:
            bce_loss = self.focal_loss(predicted_logits, true_labels)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                predicted_logits, true_labels, weight=self.class_weight
            )

        l1_regularization = torch.mean(
            torch.abs(predicted_importance - original_importance)
        )

        loss = self.bce_weight * bce_loss
        if self.use_l1_reg:
            loss += self.l1_regularization_weight * l1_regularization

        return loss

    def focal_loss(
        self, predicted_logits: torch.Tensor, true_labels: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            predicted_logits, true_labels, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.focal_loss_gamma * bce_loss

        if self.class_weight is not None:
            focal_loss = self.class_weight[true_labels.long()] * focal_loss

        return focal_loss.mean()
