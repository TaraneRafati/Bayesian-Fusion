import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, pos_weight=1.0, weight=None, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()

        if isinstance(pos_weight, list):
            self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        elif isinstance(pos_weight, (float, int)):
            self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        elif isinstance(pos_weight, torch.Tensor):
            self.pos_weight = pos_weight
        else:
            raise TypeError(f"Unsupported pos_weight type: {type(pos_weight)}")

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        pos_weight = self.pos_weight.to(inputs.device)

        weight = self.weight
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            weight = weight.to(inputs.device)

        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            weight=weight,
            reduction=self.reduction
        )

        return loss_fn(inputs, targets)
