# This file contains test problems and utility functions for synthetic preference learning experiments.
# All code here credit to Jerry Lin.
# https://github.com/facebookresearch/preference-exploration/blob/main/test_functions.py

import torch
# from botorch.distributions.distributions import Kumaraswamy # TODO: fix this later
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import DTLZ2, VehicleSafety
from torch import Tensor

# probit noise such that the DM makes 10% error for top 10% utilty using random X
probit_noise_dict = {
    "vehiclesafety_5d3d_kumaraswamyproduct": 0.0203,
    "dtlz2_8d4d_negl1dist": 0.0467,
    "osy_6d8d_piecewiselinear": 2.4131,
    "carcabdesign_7d9d_piecewiselinear": 0.1151,
    "vehiclesafety_5d3d_piecewiselinear": 0.1587,
    "dtlz2_8d4d_piecewiselinear": 0.1872,
    "osy_6d8d_sigmodconstraints": 0.0299,
    "carcabdesign_7d9d_linear": 0.0439,
}


# ========== Test Problems ============


class AdaptedOSY(MultiObjectiveTestProblem):
    r"""
    Adapted OSY test problem from [Oszycka1995]_.
    This is adapted from botorch implementation.
    We negated the fs and treat gs a objectives so that the goal is to maximzie everything
    """

    dim = 6
    num_objectives = 8
    _bounds = [
        (0.0, 10.0),
        (0.0, 10.0),
        (1.0, 5.0),
        (0.0, 6.0),
        (1.0, 5.0),
        (0.0, 10.0),
    ]
    # Placeholder reference point
    _ref_point = [0.0] * 8

    def evaluate_true(self, X: Tensor) -> Tensor:
        f1 = (
            25 * (X[..., 0] - 2) ** 2
            + (X[..., 1] - 2) ** 2
            + (X[..., 2] - 1) ** 2
            + (X[..., 3] - 4) ** 2
            + (X[..., 4] - 1) ** 2
        )
        f2 = -(X**2).sum(-1)
        g1 = X[..., 0] + X[..., 1] - 2.0
        g2 = 6.0 - X[..., 0] - X[..., 1]
        g3 = 2.0 - X[..., 1] + X[..., 0]
        g4 = 2.0 - X[..., 0] + 3.0 * X[..., 1]
        g5 = 4.0 - (X[..., 2] - 3.0) ** 2 - X[..., 3]
        g6 = (X[..., 4] - 3.0) ** 2 + X[..., 5] - 4.0
        return torch.stack([f1, f2, g1, g2, g3, g4, g5, g6], dim=-1)


class NegativeVehicleSafety(VehicleSafety):
    def evaluate_true(self, X: Tensor) -> Tensor:
        f = -super().evaluate_true(X)
        Y_bounds = torch.tensor(
            [
                [-1.7040e03, -1.1708e01, -2.6192e-01],
                [-1.6619e03, -6.2136e00, -4.2879e-02],
            ]
        ).to(X)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
        return f


class CarCabDesign(MultiObjectiveTestProblem):
    r"""RE9-7-1 car cab design from Tanabe & Ishibuchi (2020)"""

    dim = 7
    num_objectives = 9
    _bounds = [
        (0.5, 1.5),
        (0.45, 1.35),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.875, 2.625),
        (0.4, 1.2),
        (0.4, 1.2),
    ]
    _ref_point = [0.0, 0.0]  # TODO: Determine proper reference point

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.empty(
            X.shape[:-1] + (self.num_objectives,), dtype=X.dtype, device=X.device
        )

        X1 = X[..., 0]
        X2 = X[..., 1]
        X3 = X[..., 2]
        X4 = X[..., 3]
        X5 = X[..., 4]
        X6 = X[..., 5]
        X7 = X[..., 6]
        # # stochastic variables
        # X8 = 0.006 * (torch.randn_like(X1)) + 0.345
        # X9 = 0.006 * (torch.randn_like(X1)) + 0.192
        # X10 = 10 * (torch.randn_like(X1)) + 0.0
        # X11 = 10 * (torch.randn_like(X1)) + 0.0

        # not using stochastic variables for the real function
        X8 = torch.zeros_like(X1)
        X9 = torch.zeros_like(X1)
        X10 = torch.zeros_like(X1)
        X11 = torch.zeros_like(X1)

        # First function
        # negate the first function as we want minimize car weight
        f[..., 0] = -(
            1.98
            + 4.9 * X1
            + 6.67 * X2
            + 6.98 * X3
            + 4.01 * X4
            + 1.75 * X5
            + 0.00001 * X6
            + 2.73 * X7
        )
        # Second function
        f[..., 1] = 1 - (
            1.16
            - 0.3717 * X2 * X4
            - 0.00931 * X2 * X10
            - 0.484 * X3 * X9
            + 0.01343 * X6 * X10
        )
        # Third function
        f[..., 2] = 0.32 - (
            0.261
            - 0.0159 * X1 * X2
            - 0.188 * X1 * X8
            - 0.019 * X2 * X7
            + 0.0144 * X3 * X5
            + 0.87570001 * X5 * X10
            + 0.08045 * X6 * X9
            + 0.00139 * X8 * X11
            + 0.00001575 * X10 * X11
        )
        # Fourth function
        f[..., 3] = 0.32 - (
            0.214
            + 0.00817 * X5
            - 0.131 * X1 * X8
            - 0.0704 * X1 * X9
            + 0.03099 * X2 * X6
            - 0.018 * X2 * X7
            + 0.0208 * X3 * X8
            + 0.121 * X3 * X9
            - 0.00364 * X5 * X6
            + 0.0007715 * X5 * X10
            - 0.0005354 * X6 * X10
            + 0.00121 * X8 * X11
            + 0.00184 * X9 * X10
            - 0.018 * X2 * X2
        )
        # Fifth function
        f[..., 4] = 0.32 - (
            0.74
            - 0.61 * X2
            - 0.163 * X3 * X8
            + 0.001232 * X3 * X10
            - 0.166 * X7 * X9
            + 0.227 * X2 * X2
        )
        # SiXth function
        tmp = (
            (
                28.98
                + 3.818 * X3
                - 4.2 * X1 * X2
                + 0.0207 * X5 * X10
                + 6.63 * X6 * X9
                - 7.77 * X7 * X8
                + 0.32 * X9 * X10
            )
            + (
                33.86
                + 2.95 * X3
                + 0.1792 * X10
                - 5.057 * X1 * X2
                - 11 * X2 * X8
                - 0.0215 * X5 * X10
                - 9.98 * X7 * X8
                + 22 * X8 * X9
            )
            + (46.36 - 9.9 * X2 - 12.9 * X1 * X8 + 0.1107 * X3 * X10)
        ) / 3
        f[..., 5] = 32 - tmp
        # Seventh function
        f[..., 6] = 32 - (
            4.72
            - 0.5 * X4
            - 0.19 * X2 * X3
            - 0.0122 * X4 * X10
            + 0.009325 * X6 * X10
            + 0.000191 * X11 * X11
        )
        # Eighth function
        f[..., 7] = 4 - (
            10.58
            - 0.674 * X1 * X2
            - 1.95 * X2 * X8
            + 0.02054 * X3 * X10
            - 0.0198 * X4 * X10
            + 0.028 * X6 * X10
        )
        # Ninth function
        f[..., 8] = 9.9 - (
            16.45
            - 0.489 * X3 * X7
            - 0.843 * X5 * X6
            + 0.0432 * X9 * X10
            - 0.0556 * X9 * X11
            - 0.000786 * X11 * X11
        )

        Y_bounds = torch.tensor(
            [
                [
                    -4.2150e01,
                    -4.7829e-01,
                    -1.1563e02,
                    -7.2040e-03,
                    -1.8255e-01,
                    -1.0168e01,
                    2.7023e01,
                    -8.0731e00,
                    -6.4556e00,
                ],
                # Old upper bound from 1e8 points
                # [-16.0992,   0.9511, 112.7138,   0.2750,   0.1909,  14.4804,  28.9855, -2.4875, -0.8270],
                # make upper bounds of constraints to be something > 0 so that it's possible to not violate the constraints
                [
                    -16.0992,
                    0.9511,
                    112.7138,
                    0.2750,
                    0.1909,
                    14.4804,
                    28.9855,
                    0.5,
                    0.5,
                ],
            ]
        ).to(f)
        f = (f - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])

        # normalize f to between 0 and 1 roughly so that we won't disadvantage ParEGO
        return f


# ======= Utility functions ==========


class OSYSigmoidConstraintsUtil(torch.nn.Module):
    def __init__(self, Y_bounds):
        super().__init__()
        self.register_buffer("Y_bounds", Y_bounds)

    def calc_raw_util_per_dim(self, Y):
        Y_bounds = self.Y_bounds
        obj_Y = Y[..., :2]
        constr_Y = Y[..., 2:]
        norm_obj_Y = (obj_Y - Y_bounds[0, :2]) / (Y_bounds[1, :2] - Y_bounds[0, :2])

        obj_vals = norm_obj_Y.exp()
        constr_vals = torch.sigmoid(
            50
            * constr_Y
            / torch.min(torch.stack((-Y_bounds[0, 2:], Y_bounds[1, 2:])), dim=0).values
        )
        return torch.cat((obj_vals, constr_vals), dim=-1)

    def forward(self, Y, X=None):
        util_vals = self.calc_raw_util_per_dim(Y)
        constr_vals = util_vals[..., 2:]
        obj_vals = util_vals[..., :2]

        obj_sum = obj_vals.sum(-1)
        constr_prod = constr_vals.prod(dim=-1)

        util = obj_sum * constr_prod
        return util


class NegDist(torch.nn.Module):
    def __init__(self, ideal_point, p, square=False):
        super().__init__()
        self.register_buffer("ideal_point", ideal_point)
        self.p = p
        self.square = square

    def forward(self, Y, X=None):
        if len(Y.shape) == 1:
            Y = Y.unsqueeze(0)
        expanded_ideal = self.ideal_point.expand(Y.shape[:-2] + (1, -1)).contiguous()
        dist = torch.cdist(Y, expanded_ideal, p=self.p).squeeze(-1)
        if self.square:
            return -dist.square()
        else:
            return -dist


class LinearUtil(torch.nn.Module):
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


class PiecewiseLinear(torch.nn.Module):
    def __init__(
        self, beta1: torch.Tensor, beta2: torch.Tensor, thresholds: torch.Tensor
    ):
        """
        Args:
            beta1: size `output_dim` tensor
            beta2: size `output_dim` tensor
            thresholds: size `output_dim` tensor
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
        return util_val


# class KumaraswamyCDF(torch.nn.Module):
#     def __init__(self, concentration1, concentration2, Y_bounds):
#         super().__init__()
#         self.register_buffer("concentration1", concentration1)
#         self.register_buffer("concentration2", concentration2)
#         self.register_buffer("Y_bounds", Y_bounds)
#         self.kdist = Kumaraswamy(concentration1, concentration2)

#     def calc_raw_util_per_dim(self, Y):
#         Y_bounds = self.Y_bounds

#         Y = (Y - Y_bounds[0, :]) / (Y_bounds[1, :] - Y_bounds[0, :])
#         eps = 1e-6
#         Y = torch.clamp(Y, min=eps, max=1 - eps)

#         util_val = self.kdist.cdf(Y)

#         return util_val

#     def forward(self, Y, X=None):
#         util_val = self.calc_raw_util_per_dim(Y)
#         util_val = util_val[..., ::2] * util_val[..., 1::2]
#         util_val = util_val.sum(dim=-1)

#         return util_val


# class KumaraswamyCDFProduct(KumaraswamyCDF):
#     def forward(self, Y, X=None):
#         util_val = self.calc_raw_util_per_dim(Y)
#         util_val = torch.prod(util_val, dim=-1)

#         return util_val


class PiecewiseUtil(torch.nn.Module):
    def __init__(
        self, beta: torch.Tensor, thresholds: torch.Tensor, alphas, ymin, ymax
    ):
        super().__init__()
        self.register_buffer("beta", beta)
        self.thresholds = thresholds.to(beta)
        self.alphas = alphas
        self.shift = 1
        self.pow_size = 4
        self.ymin = ymin

        n_max = (
            self.calc_raw_util_per_dim(
                torch.full(size=(1, beta.shape[0]), fill_value=ymax).to(beta)
            )
            .max()
            .item()
        )
        n_min = (
            self.calc_raw_util_per_dim(
                torch.full(size=(1, beta.shape[0]), fill_value=ymin).to(beta)
            )
            .min()
            .item()
        )
        self.norm_range = (n_min, n_max)

    def calc_raw_util_per_dim(self, Y):
        # assuming Y is generally between 0 and 1.5
        Y = torch.clamp(Y, min=self.ymin)
        shift = self.shift
        alphas = self.alphas
        thresholds = self.thresholds
        pow_size = self.pow_size
        beta_mat = self.beta.expand(Y.shape)
        thresholds_mat = thresholds.expand(Y.shape)
        alphas_mat = alphas.expand(Y.shape)
        below_threshold = Y < self.thresholds
        util_val = torch.empty_like(Y)

        util_val[below_threshold] = (
            Y[below_threshold] - thresholds_mat[below_threshold] - shift
        ).pow(pow_size)
        util_val[below_threshold] = (
            -util_val[below_threshold] + shift**pow_size
        ) * alphas_mat[below_threshold]
        util_val[~below_threshold] = (
            Y[~below_threshold] - thresholds_mat[~below_threshold]
        ) * beta_mat[~below_threshold]

        return util_val

    def forward(self, Y, X=None):
        if len(Y.shape) == 1:
            Y = Y.unsqueeze(0)

        util_val = self.calc_raw_util_per_dim(Y)
        util_val = (util_val - self.norm_range[0]) / (
            self.norm_range[1] - self.norm_range[0]
        )
        util_val_int = util_val
        util_val_int = util_val_int[..., ::2] * util_val_int[..., 1::2]

        util_val = util_val.sum(-1) + util_val_int.sum(-1)
        return util_val


# ========== util function for setting up problem =========


def problem_setup(problem_str, noisy=False, **tkwargs):
    """example problem_str:
    "vehiclesafety_5d3d_kumaraswamyproduct"
    "dtlz2_8d4d_negl1dist"
    "osy_6d8d_piecewiselinear"
    "carcabdesign_7d9d_piecewiselinear"
    "vehiclesafety_5d3d_piecewiselinear"
    "dtlz2_8d4d_piecewiselinear"
    "osy_6d8d_sigmodconstraints"
    "carcabdesign_7d9d_linear"
    """
    problem_name, dims_str, util_type = problem_str.split("_")
    Y_bounds = None

    # dtlz 2 response surface
    if problem_name == "dtlz2":
        dims = dims_str.split("d")
        X_dim, Y_dim = int(dims[0]), int(dims[1])
        if dims_str == "8d4d":
            # upper bound obatined using 1.2 * max
            Y_bounds = torch.tensor(
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [2.5366, 2.5237, 2.5996, 2.6484],
                ]
            ).to(**tkwargs)
        else:
            raise RuntimeError("Unsupported problem_str")
        if noisy:
            # lowered noise level
            noise_std = 0.05
            # noise_std = 0.1
        else:
            noise_std = 0

        problem = DTLZ2(dim=X_dim, num_objectives=Y_dim, noise_std=noise_std).to(
            **tkwargs
        )
        # min-max normalization range for creating interaction terms
        ymin, ymax = 0.0, 1.5

        # utility functions for dtlz2
        if util_type == "piecewiselinear":
            if Y_dim == 4:
                beta1 = torch.tensor([4, 3, 2, 1]).to(**tkwargs)
                beta2 = torch.tensor([0.4, 0.3, 0.2, 0.1]).to(**tkwargs)
                thresholds = torch.tensor([1.0] * Y_dim).to(**tkwargs)
                get_util = PiecewiseLinear(
                    beta1=beta1, beta2=beta2, thresholds=thresholds
                )
            else:
                raise RuntimeError("Unsupported Y_dim for piecewise linear utility")
        elif util_type == "negl1dist":
            get_util = NegDist(
                problem.evaluate_true(torch.tensor([0.5] * X_dim, **tkwargs)),
                p=1,
                square=False,
            )
        else:
            raise RuntimeError("Unsupported utility!")
    elif problem_name == "vehiclesafety":
        if noisy:
            # lowered noise level
            noise_std = 0.05
        else:
            noise_std = 0
        # we wish to minimize all metrics in the original problems
        # hence we negate all values
        problem = NegativeVehicleSafety(noise_std=noise_std).to(**tkwargs)
        X_dim = problem.dim
        Y_dim = problem.num_objectives
        Y_bounds = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
            ]
        ).to(**tkwargs)
        if util_type == "piecewiselinear":
            beta1 = torch.tensor([2, 6, 8]).to(**tkwargs)
            beta2 = torch.tensor([1, 2, 2]).to(**tkwargs)
            thresholds = torch.tensor([0.5, 0.8, 0.8]).to(**tkwargs)
            get_util = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)
        # elif util_type == "kumaraswamyproduct":
        #     concentration1 = torch.tensor([0.5, 1, 1.5]).to(**tkwargs)
        #     concentration2 = torch.tensor([1.0, 2.0, 3.0]).to(**tkwargs)
        #     get_util = KumaraswamyCDFProduct(
        #         concentration1=concentration1,
        #         concentration2=concentration2,
        #         Y_bounds=Y_bounds,
        #     )
        else:
            raise RuntimeError("Unsupported utility!")
    elif problem_name == "carcabdesign":
        if noisy:
            # lowered noise level
            noise_std = 0.02
        else:
            noise_std = 0
        problem = CarCabDesign(noise_std=noise_std).to(**tkwargs)
        X_dim = problem.dim
        Y_dim = problem.num_objectives
        Y_bounds = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).to(**tkwargs)

        if util_type == "linear":
            beta = torch.tensor([2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25]).to(
                **tkwargs
            )
            get_util = LinearUtil(beta=beta)
        elif util_type == "piecewiselinear":
            beta1 = torch.tensor([7.0, 6.75, 6.5, 6.25, 6.0, 5.75, 5.5, 5.25, 5.0]).to(
                **tkwargs
            )
            beta2 = torch.tensor(
                [0.5, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225]
            ).to(**tkwargs)
            thresholds = torch.tensor(
                [0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47]
            ).to(**tkwargs)
            get_util = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)
        else:
            raise RuntimeError("Unsupported utility!")
    elif problem_name == "osy":
        if noisy:
            raise NotImplementedError("Noise level not yet determined!")
        else:
            noise_std = 0
        if dims_str == "6d8d":
            # Scale the empirical bounds by 1.1 to make sure we can include extreme values
            Y_bounds = torch.tensor(
                [
                    [
                        4.2358e-02,
                        -3.7138e02,
                        -1.9988e00,
                        -1.3999e01,
                        -7.9987e00,
                        -7.9990e00,
                        -5.9989e00,
                        -4.0000e00,
                    ],
                    [
                        1707.5742,
                        -2.6934,
                        17.9988,
                        5.9988,
                        11.9993,
                        31.9968,
                        3.9999,
                        9.9983,
                    ],
                ]
            ).to(**tkwargs)
        problem = AdaptedOSY(noise_std=noise_std).to(**tkwargs)
        X_dim = problem.dim
        Y_dim = problem.num_objectives

        if util_type == "piecewiselinear":
            if Y_dim == 8:
                beta1 = torch.tensor([0.02, 0.2, 10, 10, 10, 10, 10, 10]).to(**tkwargs)
                beta2 = torch.tensor([0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(
                    **tkwargs
                )
                thresholds = torch.tensor([1000, -100] + [0.0] * (Y_dim - 2)).to(
                    **tkwargs
                )
            else:
                raise RuntimeError("Unsupported Y_dim for betacdf utility")
            get_util = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)
        elif util_type == "sigmodconstraints":
            get_util = OSYSigmoidConstraintsUtil(Y_bounds=Y_bounds)
        else:
            raise RuntimeError("Unsupported problem!")

    if problem_str in probit_noise_dict:
        probit_noise = probit_noise_dict[problem_str]
    else:
        probit_noise = None
    print(f"{problem_str}, noisy: {noisy}, noise_std: {problem.noise_std}")
    return X_dim, Y_dim, problem, util_type, get_util, Y_bounds, probit_noise


class AugmentedProblem(MultiObjectiveTestProblem):
    def __init__(
        self, problem, noise, augmented_dim = None, duplicate=False, n_duplicates=None,
        problem_seed = None
    ):

        self._bounds = problem.bounds
        self._ref_point = problem._ref_point

        super().__init__()
        self.base_problem = problem
        self.dim = problem.dim
        self.base_outcome_dim = problem.num_objectives
        self.noise = noise
        self.bounds = problem.bounds
        self.duplicate = duplicate
        self.n_duplicates = n_duplicates
        if problem_seed is not None:
            self.problem_seed = problem_seed
        else:
            self.problem_seed = 1234

        torch.manual_seed(self.problem_seed)

        if self.duplicate:
            self.outcome_dim = self.base_outcome_dim * self.n_duplicates
        else:
            self.outcome_dim = augmented_dim
            self.outcome_projection_matrix = torch.randn(
                (self.base_outcome_dim, self.outcome_dim), dtype=torch.double
            )
            print("Generated outcome projection matrix: ", self.outcome_projection_matrix)

    def evaluate_true(self, X):

        base_outcome = self.base_problem.evaluate_true(X)

        if not self.duplicate:

            # Previously: concatenate base_outcome with a lot of pure noise
            # res = torch.cat(
            #     (
            #         self.base_problem.evaluate_true(X),
            #         torch.randn(
            #             (X.shape[0], int(self.outcome_dim - self.base_outcome_dim))
            #         )
            #         * self.noise,
            #     ),
            #     dim=1,
            # )

            # project base_outcome to higher-dimensional space
            res = torch.matmul(base_outcome, self.outcome_projection_matrix)
            noise = torch.randn_like(res) * self.noise
            res = res + noise
        else:
            # create and concatenate duplicates of base_outcome
            noise_added = (
                torch.randn(
                    (
                        base_outcome.shape[0],
                        base_outcome.shape[1] * (self.n_duplicates - 1),
                    )
                )
                * self.noise
            )
            noise_added = torch.cat(
                (torch.zeros_like(base_outcome), noise_added), dim=1
            )

            base_outcome_repeated = base_outcome.repeat(1, self.n_duplicates)
            res = base_outcome_repeated + noise_added

        return res


def problem_setup_augmented(
    problem_str, 
    augmented_dims_noise, 
    noisy=False, 
    problem_seed = None, 
    **tkwargs
):
    """example problem_str:
    "carcabdesign_7d9d_piecewiselinear_20"
    "vehiclesafety_5d3d_piecewiselinear_5c"
    "osy_6d8d_sigmoidconstraints_40"
    """
    problem_name, dims_str, util_type, augmented_str = problem_str.split("_")
    if augmented_str[-1] == "c":
        duplicate = True
        augmented_dim = None
        n_duplicates = int(augmented_str[:-1])
    else:
        # TODO: right now, only allows augmented dim to be multiples of original outcome dim
        duplicate = False
        augmented_dim = int(augmented_str)
        n_duplicates = None

    Y_bounds = None

    # dtlz 2 response surface
    if problem_name == "dtlz2":
        dims = dims_str.split("d")
        X_dim, Y_dim = int(dims[0]), int(dims[1])
        if dims_str == "8d4d":
            # upper bound obatined using 1.2 * max
            Y_bounds = torch.tensor(
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [2.5366, 2.5237, 2.5996, 2.6484],
                ]
            ).to(**tkwargs)
        else:
            raise RuntimeError("Unsupported problem_str")
        if noisy:
            # lowered noise level
            noise_std = 0.05
            # noise_std = 0.1
        else:
            noise_std = 0

        problem = DTLZ2(dim=X_dim, num_objectives=Y_dim, noise_std=noise_std).to(
            **tkwargs
        )
        # min-max normalization range for creating interaction terms
        ymin, ymax = 0.0, 1.5

        # utility functions for dtlz2
        if util_type == "piecewiselinear":
            if Y_dim == 4:
                beta1 = torch.tensor([4, 3, 2, 1]).to(**tkwargs)
                beta2 = torch.tensor([0.4, 0.3, 0.2, 0.1]).to(**tkwargs)
                thresholds = torch.tensor([1.0] * Y_dim).to(**tkwargs)

                get_util = PiecewiseLinear(
                    beta1=beta1, beta2=beta2, thresholds=thresholds
                )
            else:
                raise RuntimeError("Unsupported Y_dim for piecewise linear utility")
        elif util_type == "negl1dist":
            get_util = NegDist(
                problem.evaluate_true(torch.tensor([0.5] * X_dim, **tkwargs)),
                p=1,
                square=False,
            )
        else:
            raise RuntimeError("Unsupported utility!")
    elif problem_name == "vehiclesafety":
        if noisy:
            # lowered noise level
            noise_std = 0.05
        else:
            noise_std = 0
        # we wish to minimize all metrics in the original problems
        # hence we negate all values
        problem = NegativeVehicleSafety(noise_std=noise_std).to(**tkwargs)
        X_dim = problem.dim
        Y_dim = problem.num_objectives
        Y_bounds = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
            ]
        ).to(**tkwargs)
        if util_type == "piecewiselinear":
            beta1 = torch.tensor([2, 6, 8]).to(**tkwargs)
            beta2 = torch.tensor([1, 2, 2]).to(**tkwargs)
            thresholds = torch.tensor([0.5, 0.8, 0.8]).to(**tkwargs)

            get_util = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)
        else:
            raise RuntimeError("Unsupported utility!")
    elif problem_name == "carcabdesign":
        if noisy:
            # lowered noise level
            noise_std = 0.02
        else:
            noise_std = 0
        problem = CarCabDesign(noise_std=noise_std).to(**tkwargs)
        X_dim = problem.dim
        Y_dim = problem.num_objectives
        Y_bounds = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).to(**tkwargs)

        if util_type == "linear":
            beta = torch.tensor([2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25]).to(
                **tkwargs
            )
            get_util = LinearUtil(beta=beta)
        elif util_type == "piecewiselinear":
            beta1 = torch.tensor([7.0, 6.75, 6.5, 6.25, 6.0, 5.75, 5.5, 5.25, 5.0]).to(
                **tkwargs
            )
            beta2 = torch.tensor(
                [0.5, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225]
            ).to(**tkwargs)
            thresholds = torch.tensor(
                [0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47]
            ).to(**tkwargs)
            get_util = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)
        else:
            raise RuntimeError("Unsupported utility!")
    elif problem_name == "osy":
        if noisy:
            raise NotImplementedError("Noise level not yet determined!")
        else:
            noise_std = 0
        if dims_str == "6d8d":
            # Scale the empirical bounds by 1.1 to make sure we can include extreme values
            Y_bounds = torch.tensor(
                [
                    [
                        4.2358e-02,
                        -3.7138e02,
                        -1.9988e00,
                        -1.3999e01,
                        -7.9987e00,
                        -7.9990e00,
                        -5.9989e00,
                        -4.0000e00,
                    ],
                    [
                        1707.5742,
                        -2.6934,
                        17.9988,
                        5.9988,
                        11.9993,
                        31.9968,
                        3.9999,
                        9.9983,
                    ],
                ]
            ).to(**tkwargs)
        problem = AdaptedOSY(noise_std=noise_std).to(**tkwargs)
        X_dim = problem.dim
        Y_dim = problem.num_objectives

        if util_type == "piecewiselinear":
            if Y_dim == 8:
                beta1 = torch.tensor([0.02, 0.2, 10, 10, 10, 10, 10, 10]).to(**tkwargs)
                beta2 = torch.tensor([0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(
                    **tkwargs
                )
                thresholds = torch.tensor([1000, -100] + [0.0] * (Y_dim - 2)).to(
                    **tkwargs
                )
            else:
                raise RuntimeError("Unsupported Y_dim for betacdf utility")
            get_util = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)
        elif util_type == "sigmodconstraints":
            get_util = OSYSigmoidConstraintsUtil(Y_bounds=Y_bounds)
        else:
            raise RuntimeError("Unsupported problem!")

    if problem_str in probit_noise_dict:
        probit_noise = probit_noise_dict[problem_str]
    else:
        probit_noise = None
    print(f"{problem_str}, noisy: {noisy}, noise_std: {problem.noise_std}")

    augmented_problem = AugmentedProblem(
        problem=problem,
        noise=augmented_dims_noise,
        augmented_dim = augmented_dim,
        duplicate=duplicate,
        n_duplicates=n_duplicates,
        problem_seed = problem_seed
    )
    if duplicate:
        augmented_dim = augmented_problem.base_outcome_dim * n_duplicates
    # whether duplicate or not, augmented_dim = augmented_problem.outcome_dim
    if util_type == "linear":
        if not duplicate:
            beta = torch.cat((beta, torch.zeros(augmented_dim - Y_dim))).to(**tkwargs)
        else:
            beta = beta.repeat(1, n_duplicates).squeeze(0).to(**tkwargs)
        augmented_util_func = LinearUtil(beta=beta)
    elif util_type == "piecewiselinear":
        if not duplicate:
            beta1 = torch.cat((beta1, torch.zeros(augmented_dim - Y_dim))).to(**tkwargs)
            beta2 = torch.cat((beta2, torch.zeros(augmented_dim - Y_dim))).to(**tkwargs)
            thresholds = torch.cat((thresholds, torch.zeros(augmented_dim - Y_dim))).to(
                **tkwargs
            )
        else:
            beta1 = beta1.repeat(1, n_duplicates).squeeze(0).to(**tkwargs)
            beta2 = beta2.repeat(1, n_duplicates).squeeze(0).to(**tkwargs)
            thresholds = thresholds.repeat(1, n_duplicates).squeeze(0).to(**tkwargs)
        augmented_util_func = PiecewiseLinear(
            beta1=beta1, beta2=beta2, thresholds=thresholds
        )
    else:
        raise RuntimeError("Unsupported utility type for augmentation")

    return (
        X_dim,
        augmented_dim,
        augmented_problem,
        util_type,
        augmented_util_func,
        Y_bounds,
        probit_noise,
    )
