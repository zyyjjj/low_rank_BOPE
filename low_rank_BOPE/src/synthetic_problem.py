from typing import List, Tuple, Type

import numpy as np
import scipy.linalg

import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions.base import ConstrainedBaseTestProblem
from gpytorch.kernels import MaternKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions.multivariate_normal import MultivariateNormal


def generate_principal_axes(output_dim: int, num_axes: int, **tkwargs) -> torch.Tensor:
    r"""
    Generate a desired number of orthonormal basis vectors in a space of specified dimension
    which will serve as principal axes for simulation.

    Args:
        output_dim: dimension of output, e.g., the number of metrics we are jointly modeling
        num_axes: number of principal axes to generate
        **tkwargs: specifies variable dtype and device

    Returns:
        `num_axes x output_dim` tensor with orthonormal rows, each row being a principal axis
    """

    assert num_axes <= output_dim, "num_axes must not exceed output_dim"

    inducing_array = np.random.rand(output_dim, num_axes)
    basis = scipy.linalg.orth(inducing_array).transpose()

    return torch.tensor(basis, **tkwargs)


class PCATestProblem(ConstrainedBaseTestProblem):

    r"""Test problem where metric data are simulated from latent processes,
    where the number of latent processes is smaller than that of the output dimension.

    Example:

        max metric_0 s.t. metric_1 <= 0
        in the domain of [0,1]
        where (metric_0, metric_1) are generated from ground truth principal axis along (1,1)

        Problem = PCATestProblem(
            opt_config = [[0], [1]],
            initial_X = torch.Tensor([[0.1], [0.2], [0.3]]),
            bounds = torch.Tensor([[0,1]]),
            ground_truth_principal_axes = torch.Tensor([[0.707, 0.707]]),
            noise_std = 0.01,
            PC_lengthscales = torch.Tensor([1]),
            PC_scaling_factors = torch.Tensor([1]),
            dtype = torch.double
        )
    """

    def __init__(
        self,
        opt_config: Tuple[List],
        initial_X: torch.Tensor,
        bounds: torch.Tensor,
        ground_truth_principal_axes: torch.Tensor,
        noise_std: float,
        PC_lengthscales: torch.Tensor,
        PC_scaling_factors: torch.Tensor,
        simulation_kernel_cls: Type[Kernel] = MaternKernel,
        jitter: float = 0.000001,
        negate: bool = False,
        **tkwargs,
    ):
        r"""

        Args:
            opt_config: Tuple of two lists, containing indices of the metrics that serve as the objective(s) and constraint(s) respectively
            initial_X: `num_initial_points x input_dim` tensor, initial inputs to warm-start the PC-generating model
            bounds: `input_dim x 2` tensor, bounds of input domain
            ground_truth_principal_axes: `num_axes x output_dim` tensor
            noise_std: real number that represents the noise (SD) of the metrics
            PC_lengthscales: `num_PCs`-dimensional tensor, where the i-th entry specifies the kernel lengthscale for simulating the i-th PC;
                if None, set to all ones
            PC_scaling_factors: `num_PCs`-dimensional tensor, where the i-th entry is the factor by which to scale the simulated PC before projecting to metric space;
                if None, set to all ones
            simulation_kernel_cls: type of kernel to use for the PCs; default is MaternKernel()
            jitter: small real number to add to the diagonal of the simulated covariance matrix to prevent non-PSDness
            negate: if True, minimize the objective; if False (default), maximize the objective
            **tkwargs: specifies variable dtype and device
        """

        self._bounds = bounds
        self.input_dim = bounds.shape[0]
        super().__init__(noise_std=noise_std, negate=negate)

        self.opt_config = opt_config
        self.ground_truth_principal_axes = ground_truth_principal_axes
        self.PC_scaling_factors = PC_scaling_factors
        self.tkwargs = tkwargs

        num_axes = ground_truth_principal_axes.shape[0]
        num_initial_points = initial_X.shape[0]

        initial_PCs = torch.Tensor().to(**tkwargs)

        for i in range(num_axes):
            kernel = simulation_kernel_cls()
            kernel.lengthscale = PC_lengthscales[i].item()

            covar = kernel(initial_X).evaluate() + jitter * torch.ones(
                kernel(initial_X).evaluate().shape
            )

            mvn_dist = MultivariateNormal(
                torch.zeros(num_initial_points).to(**tkwargs), covar.to(**tkwargs)
            )
            one_PC = mvn_dist.sample().unsqueeze(1).to(**tkwargs)

            # generate PCs and center them (so that they have zero mean)
            one_PC -= torch.mean(one_PC, dim=0)

            initial_PCs = torch.cat((initial_PCs, one_PC), dim=1)

        # fit a GP model to the initial PC values
        gen_model_PC = SingleTaskGP(initial_X, initial_PCs).to(**tkwargs)
        gen_model_PC_mll = ExactMarginalLogLikelihood(
            gen_model_PC.likelihood, gen_model_PC
        )
        fit_gpytorch_mll(gen_model_PC_mll)

        self.gen_model_PC = gen_model_PC

    def eval_metrics_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate the metric values without noise.
        """

        PC_sim = self.gen_model_PC.posterior(X).mean.to(**self.tkwargs)

        # change the PC_sim scaling
        PC_sim = PC_sim * self.PC_scaling_factors

        metric_vals_noiseless = torch.matmul(
            PC_sim, self.ground_truth_principal_axes.to(**self.tkwargs)
        )

        return metric_vals_noiseless

    def eval_metrics_noisy(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate the metric values with noise added.
        """

        metric_vals_noiseless = self.eval_metrics_true(X)
        metric_vals_noisy = metric_vals_noiseless + self.noise_std * torch.randn_like(
            metric_vals_noiseless
        )

        return metric_vals_noisy

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Return the true values of the objective metrics.
        """

        metric_vals_noiseless = self.eval_metrics_true(X)

        if len(self.opt_config[0]) == 1:
            return metric_vals_noiseless[:, self.opt_config[0][0]]
        else:
            return metric_vals_noiseless[:, self.opt_config[0]]

    def evaluate_slack_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Return the values of the constraint metrics.
        """

        metric_vals_noiselss = self.eval_metrics_true(X)

        return metric_vals_noiselss[:, self.opt_config[1]]
