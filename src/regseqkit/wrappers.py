"""Model output conversions, applied before scoring."""

import torch


# Keep recovered counts finite in float32, including expected profiles.
MAX_LOG_COUNT = 88.0


def log1p_to_counts(values: torch.Tensor) -> torch.Tensor:
    """Recover nonnegative counts from log1p predictions.

    Parameters
    ----------
    values : torch.Tensor
        Log1p predictions, in any shape. Values above 88 are clipped before
        exponentiation to avoid float32 overflow.

    Returns
    -------
    counts : torch.Tensor
        Recovered counts with the same shape, device and dtype as values.
    """
    return values.clamp(max=MAX_LOG_COUNT).expm1().clamp(min=0)


class Log1pToCounts(torch.nn.Module):
    """Wrap a model returning log1p predictions to return counts.

    Parameters
    ----------
    model : torch.nn.Module
        A module returning a tensor of log1p predictions. Output selection,
        such as selecting scalar predictions from a tuple, must happen first.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict and convert the returned values to counts."""
        return log1p_to_counts(self.model(X))
