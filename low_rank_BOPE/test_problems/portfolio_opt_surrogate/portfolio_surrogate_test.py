import torch
from botorch.utils.sampling import draw_sobol_samples
from portfolio_surrogate import DistributionalPortfolioSurrogate, utility_risk_measures


simulator = DistributionalPortfolioSurrogate()
torch.manual_seed(0)

# test on single sample
X = torch.zeros(torch.Size([3, 1, 3]))
X[..., 0] += 0.0264
X[..., 1] += 0.9377
X[..., 2] += 0.1512
# X[..., 3:5] = torch.rand(torch.Size([3, 1, 2]))
print(X)
print(simulator.evaluate_true(X))


# test on batch samples, test utility function
X = draw_sobol_samples(
    bounds = torch.Tensor([[0,0,0], [1,1,1]]),
    n=1,
    q=30
).squeeze(0)
print('X shape', X.shape)
Y = simulator.evaluate_true(X)
print('Y shape', Y.shape)

U, S, V = torch.svd(Y)
S_squared = torch.square(S)
explained_variance = S_squared / S_squared.sum()

print('explained variance', explained_variance)

util_vals = utility_risk_measures(Y, util_func_key='mean_plus_sd')
print('Y', Y)
print('util_vals', util_vals)
