#
# This file includes different loss functions for training the neural network.
#

import torch.nn as nn
import torch
from typing import Optional
import warnings
import numpy as np

Tensor = torch.Tensor


class IntervalScore(nn.Module):
    """Computes the interval score for observations and their corresponding predicted interval.
    `loss = (q_upper - q_lower) + max((q_lower - target), 0) + max((target - q_upper), 0)`

    Args:
        target (Tensor): Ground truth values. shape = [batch_size, d0, .. dn].
        q_lower (Tensor): The predicted left/lower quantile. shape = [batch_size, d0, .. dn].
        q_upper (Tensor): The predicted right/upper quantile. shape = [batch_size, d0, .. dn].
        alpha (Float): Alpha level for (1-alpha) interval
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.

    Raises:
        ValueError: One of the input shapes does not match

    Returns:
        interval_score: 1-D float `Tensor` with shape [batch_size] or Float if reduction = True

    References:
        Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction,
        and estimation. Journal of the American Statistical Association 102(477), 359-378.
    """

    def __init__(
        self,
        alpha: Tensor,
        reduce: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ):
        super(IntervalScore, self).__init__()
        self.alpha = alpha
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, target: Tensor, q_lower: Tensor, q_upper: Tensor) -> Tensor:
        if not (target.size() == q_lower.size() == q_upper.size()):
            raise ValueError("Mismatching target and prediction shapes")
        if torch.any(q_lower > q_upper):
            warnings.warn(
                "Left quantile should be smaller than right quantile, check consistency!"
            )
        sharpness = q_upper - q_lower
        calibration = (
            (
                torch.clamp_min(q_lower - target, min=0)
                + torch.clamp_min(target - q_upper, min=0)
            )
            * 2
            / self.alpha
        )
        interval_score = sharpness + calibration
        if not self.reduce:
            return interval_score
        else:
            if self.reduction == "sum":
                return torch.sum(interval_score)
            else:
                return torch.mean(interval_score)


class QuantileScore(nn.Module):
    """Computes the qauntile score (pinball loss) between `y_true` and `y_pred`.
    `loss = maximum(alpha * (y_true - y_pred), (alpha - 1) * (y_true - y_pred))`

    Args:
        target (Tensor): Ground truth values. Shape = [batch_size, d0, .. dn].
        prediction (Tensor): The predicted values. Shape = [batch_size, d0, .. dn].
        alpha: Float in [0, 1] or a tensor taking values in [0, 1] and Shape = [d0,..., dn].
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.

    Raises:
        ValueError: If input and target size do not match.

    Returns:
        quantile_score: 1-D float `Tensor` with shape [batch_size] or Float if reduction = True

    References:
      - https://en.wikipedia.org/wiki/Quantile_regression
    """

    def __init__(
        self,
        alpha: Tensor,
        reduce: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ):
        super(QuantileScore, self).__init__()
        self.alpha = alpha
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if not (target.size() == prediction.size()):
            raise ValueError(
                "Using a target size ({}) that is different to the input size ({}). "
                "".format(target.size(), prediction.size())
            )
        errors = target - prediction
        quantile_score = torch.max((self.alpha - 1) * errors, self.alpha * errors)
        if not self.reduce:
            return quantile_score
        else:
            if self.reduction == "sum":
                return torch.sum(quantile_score)
            else:
                return torch.mean(quantile_score)


class NormalCRPS(nn.Module):
    """Computes the continuous ranked probability score (CRPS) for a predictive normal distribution and corresponding observations.

    Args:
        observation (Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
        mu (Tensor): Predicted mu of normal distribution. Shape = [batch_size, d0, .. dn].
        sigma (Tensor): Predicted sigma of normal distribution. Shape = [batch_size, d0, .. dn].
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.
    Raises:
        ValueError: If sizes of target mu and sigma don't match.

    Returns:
        quantile_score: 1-D float `Tensor` with shape [batch_size] or Float if reduction = True

    References:
      - Gneiting, T. et al., 2005: Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation. Mon. Wea. Rev., 133, 1098–1118
    """

    def __init__(
        self,
        reduce: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, observation: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        if not (mu.size() == sigma.size() == observation.size()):
            raise ValueError("Mismatching target and prediction shapes")
        # Use absolute value of sigma
        sigma = torch.abs(sigma)
        loc = (observation - mu) / sigma
        Phi = 0.5 * (1 + torch.special.erf(loc / np.sqrt(2.0)))
        phi = 1 / (np.sqrt(2.0 * np.pi)) * torch.exp(-torch.pow(loc, 2) / 2.0)
        crps = sigma * (loc * (2.0 * Phi - 1) + 2.0 * phi - 1 / np.sqrt(np.pi))
        if not self.reduce:
            return crps
        else:
            if self.reduction == "sum":
                return torch.sum(crps)
            else:
                return torch.mean(crps)


class EnergyScore(nn.Module):
    """Calculates the EnergyScore

    Args:
        observation (Tensor): Observed outcome. Shape = [batch_size, d, 1]
        prediction (Tensor): Samples from predictive distribution. Shape = [batch_size, d, n_samples]
    """

    def __init__(
        self,
        reduce: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, observation: Tensor, prediction: Tensor) -> Tensor:
        # Check shapes
        if not (observation.size()[0:-1] == prediction.size()[0:-1]):
            raise ValueError("Mismatching target and prediction shapes")
        # Define dimensions
        n_samples = prediction.size()[-1]

        # Calculate terms
        es_12 = torch.sum(
            torch.sqrt(
                torch.clamp(
                    torch.matmul(torch.transpose(observation, 1, 2), observation)
                    + torch.pow(torch.linalg.norm(prediction, dim=1, keepdim=True), 2)
                    - 2 * torch.matmul(torch.transpose(observation, 1, 2), prediction),
                    min=1e-7,
                    max=1e10,
                )
            ),
            dim=(1, 2),
        )
        G = torch.matmul(torch.transpose(prediction, 1, 2), prediction)
        d = torch.unsqueeze(torch.diagonal(G, dim1=1, dim2=2), dim=1)

        es_22 = torch.sum(
            torch.sqrt(
                torch.clip(d + torch.transpose(d, 1, 2) - 2 * G, min=1e-7, max=1e10)
            ),
            dim=(1, 2),
        )

        score = es_12 / (n_samples) - es_22 / (2 * n_samples * (n_samples - 1))

        if not self.reduce:
            return score
        else:
            if self.reduction == "sum":
                return torch.sum(score)
            else:
                return torch.mean(score)
