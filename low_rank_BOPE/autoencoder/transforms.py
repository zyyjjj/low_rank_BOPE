#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Optional, Tuple
import torch
import torch.nn
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior, TransformedPosterior


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
