from typing import List, Optional, Tuple, Type

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


def generate_principal_axes(output_dim: int, num_axes: int, seed: Optional[int] = None, **tkwargs) -> Tensor:
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

    if seed is None:
        seed = 1234
    np.random.seed(seed)

    inducing_array = np.random.rand(output_dim, num_axes)
    basis = scipy.linalg.orth(inducing_array).transpose()

    return torch.tensor(basis, **tkwargs)


def make_controlled_coeffs(
    full_axes: Tensor, 
    latent_dim: int, 
    alpha: float, 
    n_reps: int, 
    seed: Optional[int] = None, 
    **tkwargs
):
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
        n_reps: number of coefficient vectors to generate
    Returns:
        `n_reps x outcome_dim` tensor, with each row being a linear
            utility function coefficient
    """

    k = full_axes.shape[0]
    if seed is None:
        seed=1234
    torch.manual_seed(seed)

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
            true_axes = torch.Tensor([[0.707, 0.707]]),
            noise_std = 0.01,
            PC_lengthscales = torch.Tensor([1]),
            PC_scaling_factors = torch.Tensor([1]),
            dtype = torch.double
        )
    """

    def __init__(
        self,
        opt_config: Tuple[List],
        initial_X: Tensor,
        bounds: Tensor,
        true_axes: Tensor,
        noise_std: float,
        PC_lengthscales: Tensor,
        PC_scaling_factors: Tensor,
        simulation_kernel_cls: Type[Kernel] = MaternKernel,
        add_noise_to_PCs: bool = False,
        jitter: float = 0.000001,
        negate: bool = False,
        state_dict_str: Optional[str] = None,
        **tkwargs,
    ):
        r"""

        Args:
            opt_config: Tuple of two lists, containing indices of the metrics
                that serve as the objective(s) and constraint(s) respectively
            initial_X: `num_initial_points x input_dim` tensor, initial inputs 
                to warm-start the PC-generating model
            bounds: `input_dim x 2` tensor, bounds of input domain
            true_axes: `num_axes x output_dim` tensor
            noise_std: real number that represents the noise (SD) of the metrics
            PC_lengthscales: `num_PCs`-dimensional tensor, where the i-th entry 
                specifies the kernel lengthscale for simulating the i-th PC;
                if None, set to all ones
            PC_scaling_factors: `num_PCs`-dimensional tensor, where the i-th 
                entry is the factor by which to scale the simulated PC before 
                projecting to metric space; if None, set to all ones
            simulation_kernel_cls: type of kernel to use for the PCs; default 
                is MaternKernel()
            add_noise_to_PCs: if True, add independent noise to simulated PCs 
                (so that the noise, when projected to the outcome space, becomes 
                correlated across outcomes); if False, add independent noise 
                directly to the outcomes. Default is False.
            jitter: small real number to add to the diagonal of the simulated 
                covariance matrix to prevent non-PSDness
            negate: if True, minimize the objective; if False (default), 
                maximize the objective
            **tkwargs: specifies variable dtype and device
        """

        self._bounds = bounds
        self.input_dim = bounds.shape[0]
        self.dim = bounds.shape[0]
        super().__init__(noise_std=noise_std, negate=negate)

        self.opt_config = opt_config
        self.true_axes = true_axes
        self.PC_scaling_factors = PC_scaling_factors
        self.tkwargs = tkwargs
        self.outcome_dim = true_axes.shape[-1]
        self.add_noise_to_PCs = add_noise_to_PCs

        self.latent_dim = true_axes.shape[0]
        num_initial_points = initial_X.shape[0]

        initial_PCs = Tensor().to(**tkwargs)

        for i in range(self.latent_dim):
            kernel = simulation_kernel_cls()
            kernel.lengthscale = PC_lengthscales[i].item()

            covar_ = kernel(initial_X).evaluate()
            covar = covar_ + jitter * torch.ones(
                covar_.shape
            )

            mvn_dist = MultivariateNormal(
                torch.zeros(num_initial_points).to(
                    **tkwargs), covar.to(**tkwargs)
            )
            one_PC = mvn_dist.sample().unsqueeze(1).to(**tkwargs)

            # generate PCs and center them (so that they have zero mean)
            one_PC -= torch.mean(one_PC, dim=0)

            initial_PCs = torch.cat((initial_PCs, one_PC), dim=1)

        # fit a GP model to the initial PC values or load a saved model
        gen_model_PC = SingleTaskGP(initial_X, initial_PCs).to(**tkwargs)
        if state_dict_str is not None:
            try:
                state_dict = torch.load(f'/home/yz685/low_rank_BOPE/low_rank_BOPE/test_problems/real_metric_corr/{state_dict_str}.pt')
                gen_model_PC.load_state_dict(state_dict)
                print(f"=== Loading saved state dict for {state_dict_str} ===")
            except FileNotFoundError:
                gen_model_PC_mll = ExactMarginalLogLikelihood(
                    gen_model_PC.likelihood, gen_model_PC
                )
                from time import time
                start = time()
                fit_gpytorch_mll(gen_model_PC_mll)
                print("Fitting took %s seconds" % (time() - start))
                torch.save(
                    gen_model_PC.state_dict(), 
                    f'/home/yz685/low_rank_BOPE/low_rank_BOPE/test_problems/real_metric_corr/{state_dict_str}.pt'
                )
        else:
            gen_model_PC_mll = ExactMarginalLogLikelihood(
                gen_model_PC.likelihood, gen_model_PC
            )
            fit_gpytorch_mll(gen_model_PC_mll)

        self.gen_model_PC = gen_model_PC

    def eval_metrics_true(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the metric values without noise.
        """

        PC_sim = self.gen_model_PC.posterior(X).mean.to(**self.tkwargs)

        # change the PC_sim scaling
        PC_sim = PC_sim * self.PC_scaling_factors

        metric_vals_noiseless = torch.matmul(
            PC_sim, self.true_axes.to(**self.tkwargs)
        )

        return metric_vals_noiseless

    def eval_metrics_noisy(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the metric values with noise added.
        """

        metric_vals_noiseless = self.eval_metrics_true(X)

        if self.add_noise_to_PCs:
            PC_noise = self.noise_std * torch.randn_like()
            pass # TODO
        else:
            metric_vals_noisy = metric_vals_noiseless + \
                self.noise_std * torch.randn_like(metric_vals_noiseless)

        return metric_vals_noisy

    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Return the true values of the objective metrics.
        """

        metric_vals_noiseless = self.eval_metrics_true(X)

        if len(self.opt_config[0]) == 1:
            return metric_vals_noiseless[:, self.opt_config[0][0]]
        else:
            return metric_vals_noiseless[:, self.opt_config[0]]
    
    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""
        Evaluate the objective metrics with noise
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        latent_vars_shape = list(f.shape)
        latent_vars_shape[-1] = self.latent_dim # TODO: double check

        if noise and self.noise_std is not None:
            if self.add_noise_to_PCs:
                # generate indep noise on PC and project to outcome space
                noise = torch.matmul(
                    # torch.randn_like(self.gen_model_PC.posterior(X).mean), 
                    torch.randn(latent_vars_shape), # TODO: double check shape correct
                    self.true_axes.to(**self.tkwargs)
                )
                if len(self.opt_config[0]) == 1:
                    noise = noise[..., self.opt_config[0][0]]
                else:
                    noise = noise[..., self.opt_config[0]]
            else:
                # generate indep noise on outcomes directly
                noise = torch.randn_like(f)
            f = f + self.noise_std * noise
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)


    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""
        Return the values of the constraint metrics.
        """

        metric_vals_noiselss = self.eval_metrics_true(X)

        return metric_vals_noiselss[:, self.opt_config[1]]
    
    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""
        Evaluate constraint slack with noise.
        """
        # NOTE: we don't actually use it because we usually set constraints to 
        # be empty, but putting correlated noise case here for completeness

        cons = self.evaluate_slack_true(X=X)
        latent_vars_shape = list(cons.shape)
        latent_vars_shape[-1] = self.latent_dim # TODO: double check

        if noise and self.noise_std is not None:
            if self.add_noise_to_PCs:
                # generate indep noise on PC and project to outcome space
                noise = torch.matmul(
                    # torch.randn_like(self.gen_model_PC.posterior(X).mean), 
                    torch.randn(latent_vars_shape), # TODO: double check shape correct
                    self.true_axes.to(**self.tkwargs)
                )
                if len(self.opt_config[1]) == 1:
                    noise = noise[..., self.opt_config[1][0]]
                else:
                    noise = noise[..., self.opt_config[1]]                
            else:
                # generate indep noise on outcomes directly
                noise = torch.randn_like(cons)
            cons = cons + self.noise_std * noise

        return cons



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
        "true_axes": Tensor([1]*20),
        "PC_lengthscales": [0.1],
        "PC_scaling_factors": [2],
        "dtype": torch.double,
        "problem_seed": 1234,
    }

    # if input sth like 
    # PTS=6_input=1_outcome=45_latent=3_PCls=0.5_seed=1234
    # Rdm_input=3_outcome=50_latent=3_PCls=0.1_PCsf=2_seed=1234
    # load directly state_dict

    # overwrite config settings with kwargs
    for key, val in kwargs.items():
        if val is not None:
            config[key] = val

    np.random.seed(config["problem_seed"])
    torch.manual_seed(config["problem_seed"])
    torch.autograd.set_detect_anomaly(True)

    initial_X = torch.randn(
        (config["num_initial_samples"], config["input_dim"]), dtype=config["dtype"])

    obj_indices = list(range(config["outcome_dim"]))
    cons_indices = []

    if len(config['true_axes'].shape) == 1:
        config['true_axes'] = config['true_axes'].unsqueeze(0)

    problem = PCATestProblem(
        opt_config=(obj_indices, cons_indices),
        initial_X=initial_X,
        bounds=Tensor([[0, 1]] * config["input_dim"]),
        true_axes=config['true_axes'],
        noise_std=config["noise_std"],
        PC_lengthscales=Tensor(config["PC_lengthscales"]),
        PC_scaling_factors=Tensor(config["PC_scaling_factors"]),
        dtype=torch.double,
        state_dict_str=config.get("state_dict_str", None)
    )

    return problem


# ========== Synthetic utility functions ==========

class LinearUtil(torch.nn.Module):
    """ 
    Create linear utility function modulew with specified coefficient beta.
    f(y) = beta_1 * y_1 + ... + beta_k * y_k
    """
    def __init__(self, beta: Tensor):
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


class PiecewiseLinear(torch.nn.Module):
    def __init__(
        self, beta1: torch.Tensor, beta2: torch.Tensor, thresholds: torch.Tensor
    ):
        """
        Args:
            beta1: size `outcome_dim` tensor, coefficients below threshold (usually steep)
            beta2: size `outcome_dim` tensor, coefficients above threshold (usually flat)
            thresholds: size `outcome_dim` tensor, points where slopes change
        """
        super().__init__()
        self.register_buffer("beta1", beta1)
        self.register_buffer("beta2", beta2)
        self.register_buffer("thresholds", thresholds)

    def calc_raw_util_per_dim(self, Y):
        # below thresholds
        bt = Y < self.thresholds
        b1 = self.beta1.expand(Y.shape)
        b2 = self.beta2.expand(Y.shape)
        shift = (b2 - b1) * self.thresholds
        util_val = torch.empty_like(Y)

        # util_val[bt] = Y[bt] * b1[bt]
        util_val[bt] = Y[bt] * b1[bt] + shift[bt]
        util_val[~bt] = Y[~bt] * b2[~bt]

        return util_val

    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = util_val.sum(dim=-1)
        return util_val.unsqueeze(1)

class SumOfSquaresUtil(torch.nn.Module):
    """ 
    Create sum of squares utility function module with specified coefficient beta.
    f(y) = beta_1 * y_1^2 + ... + beta_k * y_k^2
    """
    def __init__(self, beta: Tensor):
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
    