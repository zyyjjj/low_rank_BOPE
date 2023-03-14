import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.transforms.input import (InputTransform,
                                             ReversibleInputTransform)
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior, TransformedPosterior
from torch import Tensor


# code credit to Sait
class ModifiedTransformedPosterior(TransformedPosterior):
    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample)."""
        return self.rsample().shape[-2:]

    def _extended_shape(
        self, sample_shape: torch.Size = torch.Size()  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given `sample_shape`.

        NOTE: This assumes that the `sample_transform` does not change the
        shape of the samples.
        """

        return self.rsample().shape[-2:]


# referred to Wesley's private PCA-GP code
class PCAOutcomeTransform(OutcomeTransform):
    def __init__(
        self,
        variance_explained_threshold: float = 0.9,
        num_axes: Optional[int] = None,
        *tkwargs,
    ):
        r"""
        Initialize PCAOutcomeTransform() instance
        Args:
            variance_explained_threshold: fraction of variance in the data that we want the selected principal axes to explain;
                if num_axes is None, use this to decide the number of principal axes to select
            num_axes: number of principal axes to select
        """

        super().__init__()
        self.variance_explained_threshold = variance_explained_threshold
        self.num_axes = num_axes

    def forward(
        self, Y: torch.Tensor, Yvar: Optional[torch.Tensor] = None, **tkwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Perform PCA on data Y and transforms it to a lower-dimensional representation in terms of principal components.
        Args:
            Y: `batch_shape x num_samples x output_dim` tensor of metric observations;
                assume that it is centered already, since we would normally chain the PCAOutcomeTransform() after a Standardize()
            Yvar: (optional) `batch_shape x num_samples x output_dim` tensor of metric noises (variance)
        Returns:
            Y_transformed: `batch_shape x num_samples x PC_dim` tensor of PC values
            Yvar_transformed: `batch_shape x num_samples x PC_dim` tensor of estimated PC variances
        """

        if self.training:

            U, S, V = torch.svd(Y)
            S_squared = torch.square(S)
            explained_variance = S_squared / S_squared.sum()

            if self.num_axes is None:
                # decide the number of principal axes to keep (that makes explained variance exceed the specified threshold)
                exceed_thres = (
                    np.cumsum(explained_variance) > self.variance_explained_threshold
                )
                self.num_axes = len(exceed_thres) - sum(exceed_thres) + 1

            axes_learned = torch.transpose(V[:, : self.num_axes], -2, -1)
            self.PCA_explained_variance = sum(explained_variance[: self.num_axes])
            self.axes_learned = torch.tensor(axes_learned, **tkwargs)

        Y_transformed = torch.matmul(Y, torch.transpose(self.axes_learned, -2, -1)).to(
            **tkwargs
        )

        # if Yvar is given, the variance of PCs is lower bounded by the linear combination of Yvar terms
        # technically, the variance of PCs should include the covariance between Y's, but that is usually not available
        if Yvar is not None:
            Yvar_transformed = torch.matmul(
                Yvar, torch.square(torch.transpose(self.axes_learned, -2, -1))
            ).to(**tkwargs)

        return Y_transformed, Yvar_transformed if Yvar is not None else None

    def untransform(
        self, Y: torch.Tensor, Yvar: Optional[torch.Tensor] = None, **tkwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Transform PC back to metrics according to self.axes_learned.
        Args:
            Y: `num_samples x PC_dim` tensor of PC values
            Yvar: `num_samples x PC_dim` tensor of PC variances
        Returns:
            Y_untransformed: `num_samples x output_dim` tensor of metric values
            Yvar_untransformed: `num_samples x output_dim` tensor of metric variances
        """

        Y_untransformed = torch.matmul(Y, self.axes_learned)
        if Yvar is not None:
            Yvar_untransformed = torch.matmul(Yvar, torch.square(self.axes_learned))

        return (
            Y_untransformed,
            Yvar_untransformed if Yvar is not None else None,
        )

    def untransform_posterior(self, posterior: Posterior):
        r"""
        Create posterior distribution in the space of metrics.
        Args:
            posterior: posterior in the space of PCs
        Returns:
            untransformed_posterior: posterior in the space of metrics
        """

        untransformed_posterior = ModifiedTransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.matmul(self.axes_learned),
            mean_transform=lambda x, v: x.matmul(self.axes_learned),
            variance_transform=lambda x, v: v.matmul(torch.square(self.axes_learned)),
        )

        return untransformed_posterior


class PCAInputTransform(InputTransform, torch.nn.Module):
    def __init__(
        self,
        axes: torch.Tensor,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ):
        r"""
        Initialize PCAInputTransform() instance.
        Args:
            axes: `num_axes x input_dim` tensor with norm-1 orthogonal rows
                (in the case of PE, these are the principal axes
                learned from the previous stage of fitting outcome model)
            transform_on_train: A boolean indicating whether to apply the
                transform in train() mode.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode.
            transform_on_fantasize: A boolean indicating whether to apply
                the transform when called from within a `fantasize` call.
        """
        super().__init__()
        self.axes = axes
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Transform the input X into latent representations using self.axes.
        Args:
            X: `num_samples x input_dim` tensor of input data
        """

        transformed_X = torch.matmul(X, torch.transpose(self.axes, -2, -1))

        return transformed_X

    def untransform(self, X_tf: torch.Tensor) -> torch.Tensor:
        r"""
        Untransform a latent representation back to input space.
        Args:
            X_tf: `num_samples x num_axes` tensor of latent representations
        """

        untransformed_X = torch.matmul(X_tf, self.axes)

        return untransformed_X

class LinearProjectionOutcomeTransform(OutcomeTransform):
    def __init__(
        self,
        projection_matrix: torch.Tensor,
        *tkwargs,
    ):
        r"""
        Initialize LinearProjectionOutcomeTransform() instance.
        Args:
            projection_matrix: `p x outcome_dim` tensor;
                when applied to an outcome vector, transforms it into a `p`-dimensional vector
        """

        super().__init__()
        self.projection_matrix = projection_matrix
        self.projection_matrix_pseudo_inv = torch.linalg.pinv(
            torch.transpose(projection_matrix, -2, -1)
        )

    def forward(
        self, Y: torch.Tensor, Yvar: Optional[torch.Tensor] = None, **tkwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Apply linear projection to Y and project it to `p` dimensions.
        Args:
            Y: `batch_shape x num_samples x outcome_dim` tensor of metric observations;
            Yvar: (optional) `batch_shape x num_samples x outcome_dim` tensor of metric noises (variance)
        Returns:
            Y_transformed: `batch_shape x num_samples x p` tensor of linearly projected values
            Yvar_transformed: `batch_shape x num_samples x p` tensor of linearly projected values
        """

        Y_transformed = torch.matmul(
            Y, torch.transpose(self.projection_matrix, -2, -1)
        ).to(**tkwargs)

        # TODO: Think about how to deal with correlation in the projected values

        # if Yvar is given, the variance of projections is lower bounded by the linear combination of Yvar terms
        # technically, this should also include the covariance between Y's, but that is usually not available
        if Yvar is not None:
            Yvar_transformed = torch.matmul(
                Yvar, torch.square(torch.transpose(self.projection_matrix, -2, -1))
            ).to(**tkwargs)

        return Y_transformed, Yvar_transformed if Yvar is not None else None

    def untransform(
        self, Y: torch.Tensor, Yvar: Optional[torch.Tensor] = None, **tkwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Transform projected values back to the original outcome space
        using the pseudo-inverse of the projection matrix.
        Args:
            Y: `num_samples x p` tensor of projected values
            Yvar: `num_samples x p` tensor of projected variances
        Returns:
            Y_untransformed: `num_samples x outcome_dim` tensor of outcome values
            Yvar_untransformed: `num_samples x outcome_dim` tensor of outcome variances
        """

        Y_untransformed = torch.matmul(Y, self.projection_matrix_pseudo_inv)
        if Yvar is not None:
            Yvar_untransformed = torch.matmul(
                Yvar, torch.square(self.projection_matrix_pseudo_inv)
            )

        return (
            Y_untransformed,
            Yvar_untransformed if Yvar is not None else None,
        )

    def untransform_posterior(self, posterior: Posterior):
        r"""
        Transform a posterior distribution in the projected space back to
        a posterior distribution in the original outcome space.
        Args:
            posterior: posterior in the space of projected values
        Returns:
            untransformed_posterior: posterior in the space of outcomes
        """

        untransformed_posterior = ModifiedTransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.matmul(self.projection_matrix_pseudo_inv),
            mean_transform=lambda x, v: x.matmul(self.projection_matrix_pseudo_inv),
            variance_transform=lambda x, v: v.matmul(
                torch.square(self.projection_matrix_pseudo_inv)
            ),  # TODO: think about this later
        )

        return untransformed_posterior


class LinearProjectionInputTransform(InputTransform, torch.nn.Module):
    def __init__(
        self,
        projection_matrix: torch.Tensor,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ):
        r"""
        Initialize LinearProjectionInputTransform() instance.
        Args:
            projection_matrix: `p x input_dim` tensor;
                when applied to an input vector, transforms it into a `p`-dimensional vector
            transform_on_train: A boolean indicating whether to apply the
                transform in train() mode.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode.
            transform_on_fantasize: A boolean indicating whether to apply
                the transform when called from within a `fantasize` call.
        """
        super().__init__()
        self.projection_matrix = projection_matrix
        self.projection_matrix_pseudo_inv = torch.linalg.pinv(
            torch.transpose(projection_matrix, -2, -1)
        )
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Apply linear projection to X and project it to `p` dimensions.
        Args:
            X: `num_samples x input_dim` tensor of input data
        """

        transformed_X = torch.matmul(X, torch.transpose(self.projection_matrix, -2, -1))

        return transformed_X

    def untransform(self, X_tf: torch.Tensor) -> torch.Tensor:
        r"""
        Untransform projected values back to input space.
        Args:
            X_tf: `num_samples x p` tensor of projected values
        """

        untransformed_X = torch.matmul(X_tf, self.projection_matrix_pseudo_inv)

        return untransformed_X


def generate_random_projection(dim, num_axes, **tkwargs):
    r"""
    Generate a random linear projection matrix.
    Args:
        dim: dimensionality of the full space
        num_axes: dimensionality of the projected values
    Returns:
        proj_matrix: `num_axes x dim` shape tensor with normalized rows
    """

    proj_matrix = torch.randn((num_axes, dim))
    proj_matrix = torch.nn.functional.normalize(proj_matrix, p=2, dim=1)

    return proj_matrix.to(**tkwargs)


# deprecated
def generate_subset_projection(dim: int, num_axes: int, **tkwargs):
    r"""
    Generate a linear projection onto a subset of canonical axes selected uniformly at random.
    Args:
        dim: dimensionality of the full space
        num_axes: number of canonical axes to project onto. Must be <= dim.
    Returns:
        proj_matrix: `num_axes x dim` shape tensor
        canon_set: list of selected axes
    """

    proj_matrix = torch.zeros((num_axes, dim))

    # sample `num_axes` indices uniformly at random
    canon_set = random.sample(range(dim), num_axes)

    for i in range(num_axes):
        proj_matrix[i, canon_set[i]] = 1.0

    return proj_matrix.to(**tkwargs), canon_set


class SubsetOutcomeTransform(OutcomeTransform):
    def __init__(
        self,
        outcome_dim: int,
        subset: List[int],
        *tkwargs,
    ):
        r"""
        Initialize SubsetOutcomeTransform() instance.
        (This transform picks a subset of the outcomes.)
        Args:
            outcome_dim: full outcome dimensionality
            subset: list with p entries, a subset from {1, ..., outcome_dim}
        """

        super().__init__()
        self.outcome_dim = outcome_dim
        self.subset = subset

    def forward(
        self, Y: torch.Tensor, Yvar: Optional[torch.Tensor] = None, **tkwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Select the subset of Y.
        Args:
            Y: `batch_shape x num_samples x outcome_dim` tensor of outcome observations;
            Yvar: (optional) `batch_shape x num_samples x outcome_dim` tensor of outcome noises (variance)
        Returns:
            Y_transformed: `batch_shape x num_samples x p` tensor of subset outcome values
            Yvar_transformed: `batch_shape x num_samples x p` tensor of subset outcome variances
        """

        Y_transformed = Y[..., self.subset].to(**tkwargs)

        if Yvar is not None:
            Yvar_transformed = Yvar[..., self.subset].to(**tkwargs)

        return Y_transformed, Yvar_transformed if Yvar is not None else None

    def untransform(
        self, Y: torch.Tensor, Yvar: Optional[torch.Tensor] = None, **tkwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Transform subset outcomes back to the original outcome space
        by imputing the unmodeled dimensions with zeros.

        Args:
            Y: `num_samples x p` tensor of subset outcome values
            Yvar: `num_samples x p` tensor of subset outcome variances
        Returns:
            Y_untransformed: `num_samples x outcome_dim` tensor of outcome values
            Yvar_untransformed: `num_samples x outcome_dim` tensor of outcome variances
        """

        # initialize a zero tensor with the full shapes
        # then fill in the nonzero values
        Y_untransformed = torch.zeros((*Y.shape[:-1], self.outcome_dim)).to(Y)
        Y_untransformed[..., self.subset] = Y

        if Yvar is not None:
            Yvar_untransformed = torch.zeros((*Yvar.shape[:-1], self.outcome_dim)).to(
                Yvar
            )
            Yvar_untransformed[..., self.subset] = Yvar

        return (
            Y_untransformed,
            Yvar_untransformed if Yvar is not None else None,
        )

    def untransform_posterior(self, posterior: Posterior):
        r"""
        Transform a posterior distribution on the subset of outcomes
        to a posterior distribution on the full set of outcomes.
        Args:
            posterior: posterior on the subset of outcomes
        Returns:
            untransformed_posterior: posterior on the full set of outcomes
                return zero deterministically for the unmodeled outcomes
        """

        def impute_zeros(y):
            y_untransformed = torch.zeros((*y.shape[:-1], self.outcome_dim)).to(y)
            y_untransformed[..., self.subset] = y
            return y_untransformed

        def mean_transform(y, v):
            return impute_zeros(y)

        def variance_transform(y, v):
            return impute_zeros(v)

        untransformed_posterior = ModifiedTransformedPosterior(
            posterior=posterior,
            sample_transform=impute_zeros,
            mean_transform=mean_transform,
            variance_transform=variance_transform,
        )

        return untransformed_posterior


class InputCenter(ReversibleInputTransform, torch.nn.Module):
    r"""Center the inputs (zero mean), don't change the variance.
    This class is modified from InputStandardize.

    In train mode, calling `forward` updates the module state
    (i.e. the mean/std normalizing constants). If in eval mode, calling `forward`
    simply applies the standardization using the current module state.
    """

    def __init__(
        self,
        d: int,
        indices: Optional[List[int]] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_std: float = 1e-8,
    ) -> None:
        r"""Center inputs (zero mean).

        Args:
            d: The dimension of the input space.
            indices: The indices of the inputs to standardize. If omitted,
                take all dimensions of the inputs into account.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            min_std: Amount of noise to add to the standard deviation to ensure no
                division by zero errors.
        """
        super().__init__()
        if (indices is not None) and (len(indices) == 0):
            raise ValueError("`indices` list is empty!")
        if (indices is not None) and (len(indices) > 0):
            indices = torch.tensor(indices, dtype=torch.long)
            if len(indices) > d:
                raise ValueError("Can provide at most `d` indices!")
            if (indices > d - 1).any():
                raise ValueError("Elements of `indices` have to be smaller than `d`!")
            if len(indices.unique()) != len(indices):
                raise ValueError("Elements of `indices` tensor must be unique!")
            self.indices = indices
        self.register_buffer("means", torch.zeros(*batch_shape, 1, d))
        self.register_buffer("stds", torch.ones(*batch_shape, 1, d))
        self._d = d
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.batch_shape = batch_shape
        self.min_std = min_std
        self.reverse = reverse
        self.learn_bounds = True

    def _transform(self, X: Tensor) -> Tensor:
        r"""Center the inputs.

        In train mode, calling `forward` updates the module state
        (i.e. the mean/std normalizing constants). If in eval mode, calling `forward`
        simply applies the standardization using the current module state.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of inputs, de-meaned.
        """
        if self.training and self.learn_bounds:
            if X.size(-1) != self.means.size(-1):
                raise BotorchTensorDimensionError(
                    f"Wrong input. dimension. Received {X.size(-1)}, "
                    f"expected {self.means.size(-1)}"
                )

            n = len(self.batch_shape) + 2
            if X.ndim < n:
                raise ValueError(
                    f"`X` must have at least {n} dimensions, {n - 2} batch and 2 innate"
                    f" , but has {X.ndim}."
                )

            # Aggregate means and standard deviations over extra batch and marginal dims
            batch_ndim = min(len(self.batch_shape), X.ndim - 2)  # batch rank of `X`
            reduce_dims = (*range(X.ndim - batch_ndim - 2), X.ndim - 2)
            self.stds, self.means = (
                values.unsqueeze(-2)
                for values in torch.std_mean(X, dim=reduce_dims, unbiased=True)
            )
            self.stds.clamp_(min=self.min_std)

        if hasattr(self, "indices"):
            X_new = X.clone()
            X_new[..., self.indices] = (
                X_new[..., self.indices] - self.means[..., self.indices]
            )

            return X_new

        return X - self.means

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Un-center the inputs, i.e., add back the mean.

        Args:
            X: A `batch_shape x n x d`-dim tensor of centered inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-centered inputs.
        """
        if hasattr(self, "indices"):
            X_new = X.clone()
            X_new[..., self.indices] = (
                self.means[..., self.indices] + X_new[..., self.indices]
            )

            return X_new

        return self.means.to(X) + X

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if hasattr(self, "indices") == hasattr(other, "indices"):
            if hasattr(self, "indices"):
                return (
                    super().equals(other=other)
                    and (self._d == other._d)
                    and (self.indices == other.indices).all()
                )
            else:
                return super().equals(other=other) and (self._d == other._d)
        return False


def get_latent_ineq_constraints(projection: Tensor, original_bounds: Tensor):
    """
    Get inequality constraints on latent variables

    Args:
        projection: `latent_dim x outcome_dim` tensor of projection matrix
        original_bounds: `2 x outcome_dim` tensor of bounds in the outcome space
    Returns:
        A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (latent_var[indices[i]] * coefficients[i]) >= rhs`
        (This is to be directly plugged into inequality_constraints in optimize_acqf) 
    """

    # we want: 
    # projection[0][i]*pc[0] + ... + projection[p][i]*pc[p] >= original_bounds[0][i]
    # - projection[0][i]*pc[0] - ... - projection[p][i]*pc[p] >= - original_bounds[1][i]

    indices = torch.tensor(range(projection.shape[0]))
    latent_cons = []

    for i in range(original_bounds.shape[-1]):
        lb = original_bounds[0][i].item()
        coefficients_lb = projection[:,i]
        latent_cons.append((indices, coefficients_lb, lb))

        ub = -original_bounds[1][i].item()
        coefficients_ub = -projection[:,i]
        latent_cons.append((indices, coefficients_ub, ub))
    
    return latent_cons
    

def compute_weights(util_vals: Tensor, weights_type: str, **kwargs):

    r""" 
    Compute weights for each datapoint, later used in weighted PCA.
    This function is in progress.

    Args:
        util_vals: shape `(num_samples,)` tensor of sample utility values;
            (can be true utility values, or predicted posterior mean from a surrogate model)
        weights_type: specifies the type of weights
            "rank": rank weighting in Tripp et al. 2020
            "power": 
        kwargs: settings for specific weight types

    Returns:
        weights: `num_samples x 1` tensor of weights for each data point
    """

    weights_type = "rank" if weights_type is None else weights_type

    if weights_type == "rank":
        # follows Tripp et al. paper
        k = kwargs.get("k", 10) # TODO: come back to setting k more carefully
        utils_argsort = np.argsort(-np.asarray(util_vals))
        ranks = np.argsort(utils_argsort)
        weights = 1 / (k * len(util_vals) + ranks) # TODO: should we normalize?

    # elif weights_type == "power":
    #     power = kwargs.get("power", 0)

    #     weights = np.power(np.asarray(util_vals), power) / np.sum(np.power(np.asarray(util_vals), power))
        # TODO: how to deal with negative util_val?

    return torch.tensor(weights).unsqueeze(1)



def fit_pca(train_Y: Tensor, var_threshold: float=0.9, weights: Optional[Tensor] = None):

    r"""
    Perform PCA on supplied data with optional weights.

    Args:
        train_Y: `num_samples x outcome_dim` tensor of data
        var_threshold: threshold of variance explained
        weights: `num_samples x 1` tensor of weights to add on each data point
    Returns:
        pca_axes: `latent_dim x outcome_dim` tensor where each row is a pca axis
    """

    # unweighted pca
    if weights is None:
        U, S, V = torch.svd(train_Y - train_Y.mean(dim=0))

    # weighted pca
    else:
        assert weights.shape[0] == train_Y.shape[0], \
            f"weights shape {weights.shape} does not match train_Y shape {train_Y.shape}, "
        assert (weights >= 0).all(), \
            "weights must be nonnegative"
            
        weighted_mean = (train_Y * weights).sum(dim=0) / weights.sum(0)
        train_Y_centered = train_Y - weighted_mean
        U, S, V = torch.svd(weights * train_Y_centered)

    S_squared = torch.square(S)
    explained_variance = S_squared / S_squared.sum()

    exceed_thres = (
        np.cumsum(explained_variance) > var_threshold
    )
    num_axes = len(exceed_thres) - sum(exceed_thres) + 1

    pca_axes = torch.tensor(torch.transpose(V[:, : num_axes], -2, -1), dtype = torch.double)

    return pca_axes