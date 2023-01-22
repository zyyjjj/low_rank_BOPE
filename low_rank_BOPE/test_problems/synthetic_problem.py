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
from torch import Tensor
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


def make_controlled_coeffs(full_axes, latent_dim, alpha, n_reps, **tkwargs):
    """
    Create norm-1 vectors with a specified norm in the subspace
    spanned by a specified set of axes.
    This is used here to generate coefficients for the linear
    utility function, with a controlled level of (dis)alignment
    with the subspace for outcomes.
    Args:
        full_axes: `outcome_dim x outcome_dim` orthonormal matrix,
            each row representing an axis
        latent_dim: latent dimension
        alpha: a number in [0,1] specifying the desired norm of the
            projection of each coefficient onto the space
            spanned by the first `latent_dim` rows of full_axes
        n_reps: number of coefficients to generate
    Returns:
        `n_reps x outcome_dim` tensor, with each row being a linear
            utility function coefficient
    """

    k = full_axes.shape[0]

    # first generate vectors lying in the latent space with norm alpha
    # z1 is `latent_dim x n_reps`, V1 is `outcome_dim x latent_dim`
    z1 = torch.randn((latent_dim, n_reps)).to(**tkwargs)
    V1 = torch.transpose(full_axes[:latent_dim], -2, -1).to(**tkwargs)
    Vz1 = torch.matmul(V1, z1)
    c_proj = torch.nn.functional.normalize(Vz1, dim=0) * alpha

    if alpha == 1:
        return torch.transpose(c_proj, -2, -1)

    else:
        # then generate vectors orthogonal to the latent space
        # with norm sqrt(1-alpha^2)
        # z2 is `(outcome_dim - latent_dim) x n_reps`
        # V2 is `outcome_dim x (outcome_dim - latent_dim)`
        z2 = torch.randn((k - latent_dim, n_reps)).to(**tkwargs)
        V2 = torch.transpose(
            full_axes[: (k - latent_dim)], -2, -1).to(**tkwargs)
        Vz2 = torch.matmul(V2, z2)
        c_orth = torch.nn.functional.normalize(
            Vz2, dim=0) * np.sqrt(1 - alpha**2)

        return torch.transpose(c_proj + c_orth, -2, -1)


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
            opt_config: Tuple of two lists, containing indices of the metrics
                that serve as the objective(s) and constraint(s) respectively
            initial_X: `num_initial_points x input_dim` tensor, initial inputs 
                to warm-start the PC-generating model
            bounds: `input_dim x 2` tensor, bounds of input domain
            ground_truth_principal_axes: `num_axes x output_dim` tensor
            noise_std: real number that represents the noise (SD) of the metrics
            PC_lengthscales: `num_PCs`-dimensional tensor, where the i-th entry 
                specifies the kernel lengthscale for simulating the i-th PC;
                if None, set to all ones
            PC_scaling_factors: `num_PCs`-dimensional tensor, where the i-th 
                entry is the factor by which to scale the simulated PC before 
                projecting to metric space; if None, set to all ones
            simulation_kernel_cls: type of kernel to use for the PCs; default 
                is MaternKernel()
            jitter: small real number to add to the diagonal of the simulated 
                covariance matrix to prevent non-PSDness
            negate: if True, minimize the objective; if False (default), 
                maximize the objective
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
                torch.zeros(num_initial_points).to(
                    **tkwargs), covar.to(**tkwargs)
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


def make_problem(**kwargs):
    """
    Create PCATestProblem with specified low-rank structure.
    """

    # default config
    config = {
        "input_dim": 1,
        "outcome_dim": 20,
        "PC_noise_level": 0,
        "noise_std": 0.1,
        "num_initial_samples": 20,
        "ground_truth_principal_axes": torch.Tensor([1]*20),
        "PC_lengthscales": [0.1],
        "PC_scaling_factors": [2],
        "dtype": torch.double,
        "np_seed": 1234,
        "torch_seed": 1234
    }

    # overwrite config settings with kwargs
    for key, val in kwargs.items():
        config[key] = val

    np.random.seed(config["np_seed"])
    torch.manual_seed(config["torch_seed"])
    torch.autograd.set_detect_anomaly(True)

    initial_X = torch.randn(
        (config["num_initial_samples"], config["input_dim"]), dtype=config["dtype"])

    obj_indices = list(range(config["outcome_dim"]))
    cons_indices = []

    if len(config['ground_truth_principal_axes'].shape) == 1:
        config['ground_truth_principal_axes'] = config['ground_truth_principal_axes'].unsqueeze(
            0)

    problem = PCATestProblem(
        opt_config=(obj_indices, cons_indices),
        initial_X=initial_X,
        bounds=torch.Tensor([[0, 1]] * config["input_dim"]),
        ground_truth_principal_axes=config['ground_truth_principal_axes'],
        noise_std=config["noise_std"],
        PC_lengthscales=Tensor(config["PC_lengthscales"]),
        PC_scaling_factors=Tensor(config["PC_scaling_factors"]),
        dtype=torch.double,
    )

    return problem


# ========== Synthetic utility functions ==========

class LinearUtil(torch.nn.Module):
    """ 
    Create linear utility function modulew with specified coefficient beta.
    f(y) = beta_1 * y_1 + ... + beta_k * y_k
    """
    def __init__(self, beta: torch.Tensor):
        """
        Args:
            beta: size `output_dim` tensor
        """
        super().__init__()
        self.register_buffer("beta", beta)

    def calc_raw_util_per_dim(self, Y):
        return Y * self.beta.to(Y)

    def forward(self, Y, X=None):
        return Y @ self.beta.to(Y)

class SumOfSquaresUtil(torch.nn.Module):
    """ 
    Create sum of squares utility function modulew with specified coefficient beta.
    f(y) = beta_1 * y_1^2 + ... + beta_k * y_k^2
    """
    def __init__(self, beta: torch.Tensor):
        """
        Args:
            beta: size `output_dim` tensor
        """
        super().__init__()
        self.register_buffer("beta", beta)

    def calc_raw_util_per_dim(self, Y):
        return torch.square(Y) * self.beta.to(Y)

    def forward(self, Y, X=None):
        return torch.square(Y) @ self.beta.to(Y)