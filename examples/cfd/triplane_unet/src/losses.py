import torch
import torch.nn as nn


# loss function with rel/abs Lp loss
class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class TruncatedMSELoss(nn.Module):
    def __init__(self, reduction="mean", threshold=1.0):
        super().__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input, target):
        loss = (input - target) ** 2
        loss[loss > self.threshold] = self.threshold
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss(loss_name: str = "LpLoss"):
    if loss_name == "LpLoss":
        return LpLoss(size_average=True)
    elif loss_name == "MSELoss":
        return nn.MSELoss(reduction="mean")
    elif loss_name == "TruncatedMSELoss":
        return TruncatedMSELoss(reduction="mean")
    elif loss_name == "HuberLoss":
        return nn.HuberLoss(reduction="mean")
    else:
        raise NotImplementedError(f"Loss {loss_name} not implemented")