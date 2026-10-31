"""PyTorch model components for the AMZN order-book project."""

import torch


def lagged_pairs(values, lags=1):
    """Return newest-first lag blocks and next-row targets within one segment."""
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer.")
    if values.ndim != 2 or len(values) <= lags or values.shape[1] == 0:
        raise ValueError("Need a feature matrix with more rows than lags.")
    previous = torch.cat([values[lags - lag - 1:len(values) - lag - 1] for lag in range(lags)], dim=1)
    return previous, values[lags:]


class VectorAR(torch.nn.Module):
    """VAR(p): next vector depends on p past vectors plus a sampled residual.

    Residual vectors are sampled jointly across features and independently
    across time. This preserves empirical simultaneous noise dependencies,
    but does not enforce order-book constraints or conditional volatility.
    """

    def __init__(self, weight, bias, residuals):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] < 1:
            raise ValueError("weight must be a non-empty matrix.")
        if weight.shape[1] % weight.shape[0] != 0:
            raise ValueError("weight must contain a whole number of feature-sized lag blocks.")
        size = weight.shape[0]
        if bias.shape != (size,) or residuals.ndim != 2 or residuals.shape[1] != size or len(residuals) == 0:
            raise ValueError("Incompatible bias or residual dimensions.")
        for name, value in (("weight", weight), ("bias", bias), ("residuals", residuals)):
            value = value.detach().to(device="cpu", dtype=torch.float64).clone()
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
            self.register_buffer(name, value)

    @classmethod
    @torch.no_grad()
    def fit(cls, training_features, ridge=1e-4, lags=1):
        """Fit one penalty; use VARFit directly to share work across a ridge grid."""
        return VARFit(training_features, lags).fit(ridge)

    @property
    def lags(self):
        return self.weight.shape[1] // len(self.bias)

    @property
    def spectral_radius(self):
        """Companion-matrix radius; a stability diagnostic, not book validity."""
        size = self.weight.shape[1]
        companion = self.weight.new_zeros((size, size))
        companion[:len(self.bias)] = self.weight
        if self.lags > 1:
            companion[len(self.bias):, :-len(self.bias)] = torch.eye(size - len(self.bias), dtype=self.weight.dtype,
                                                                   device=self.weight.device)
        return float(torch.linalg.eigvals(companion).abs().max())

    def forward(self, previous):
        """Mean from newest-first flat lags, or chronological (batch, p, D) history."""
        if previous.ndim == 3:
            if previous.shape[1:] != (self.lags, len(self.bias)):
                raise ValueError("History shape must match the model's lag order and features.")
            previous = previous.flip(1).flatten(1)
        if previous.shape[-1] != self.weight.shape[1]:
            raise ValueError("Input width must match the model's lag blocks.")
        return previous @ self.weight.T + self.bias

    @torch.no_grad()
    def generate(self, initial_features, sequence_length=32, seed=42):
        """Generate future updates; the supplied initial state is not returned."""
        if sequence_length < 1 or seed < 0:
            raise ValueError("sequence_length must be positive and seed non-negative.")
        history = initial_features.detach().to(device=self.weight.device, dtype=self.weight.dtype)
        if history.ndim == 2 and self.lags == 1:
            history = history[:, None, :]
        if history.ndim != 3 or history.shape[1:] != (self.lags, len(self.bias)) or len(history) == 0:
            raise ValueError("Expected (batch, lags, features) history, oldest to newest.")
        if not torch.isfinite(history).all():
            raise ValueError("Initial features contain non-finite values.")
        rng = torch.Generator(device=self.weight.device).manual_seed(seed)
        generated = history.new_empty((len(history), sequence_length, len(self.bias)))
        for step in range(sequence_length):
            indices = torch.randint(len(self.residuals), (len(history),), generator=rng, device=history.device)
            state = self(history) + self.residuals[indices]
            generated[:, step] = state
            history = torch.cat((history[:, 1:], state[:, None, :]), dim=1)
        if not torch.isfinite(generated).all():
            raise FloatingPointError("Autoregressive rollout became non-finite; no clipping was applied.")
        return generated


class VARFit:
    """One centred SVD per lag order, reused for every ridge penalty.

    For X = U diag(s) V.T, W.T = V diag(s / (s**2 + n*ridge)) U.T Y.
    This is the same least-squares objective as the augmented-system solver;
    the intercept is unpenalised and no covariance matrix is inverted.
    """

    @torch.no_grad()
    def __init__(self, training_features, lags=1):
        values = training_features.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("Training features contain non-finite values.")
        previous, following = lagged_pairs(values, lags)
        self.x_mean, self.y_mean = previous.mean(0), following.mean(0)
        self.x, self.y = previous - self.x_mean, following - self.y_mean
        u, self.s, self.vh = torch.linalg.svd(self.x, full_matrices=False)
        self.uty = u.T @ self.y  # Release tall U after construction; retain only small decomposition factors.

    @torch.no_grad()
    def fit(self, ridge=1e-4):
        if ridge < 0 or not torch.isfinite(torch.tensor(ridge)).item():
            raise ValueError("ridge must be finite and non-negative.")
        if ridge == 0:
            tolerance = torch.finfo(self.s.dtype).eps * max(self.x.shape) * self.s.max()
            factors = torch.where(self.s > tolerance, self.s.reciprocal(), torch.zeros_like(self.s))
        else:
            factors = self.s / (self.s.square() + len(self.x) * ridge)
        weight = (self.vh.T @ (factors[:, None] * self.uty)).T.contiguous()
        bias = self.y_mean - self.x_mean @ weight.T
        residuals = self.y - self.x @ weight.T
        residuals -= residuals.mean(0)
        return VectorAR(weight, bias, residuals)
