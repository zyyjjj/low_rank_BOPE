#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import torch
import torch.nn
from botorch.models.transforms.input import InputTransform


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
