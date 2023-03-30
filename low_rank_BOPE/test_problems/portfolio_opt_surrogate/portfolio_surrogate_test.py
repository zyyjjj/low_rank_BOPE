import torch
from botorch.utils.sampling import draw_sobol_samples
from portfolio_surrogate import (DistributionalPortfolioSurrogate,
                                 RiskMeasureUtil)

if __name__ == "__main__":

    simulator = DistributionalPortfolioSurrogate(
        n_w_samples=100, 
        w_bounds=torch.tensor([[0.0001, 0.0001], [0.01, 0.001]]))
    torch.manual_seed(0)

    # # test on single sample
    # X = torch.zeros(torch.Size([3, 1, 3]))
    # X[..., 0] += 0.0264
    # X[..., 1] += 0.9377
    # X[..., 2] += 0.1512
    # # X[..., 3:5] = torch.rand(torch.Size([3, 1, 2]))
    # print(X)
    # print(simulator.evaluate_true(X))


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
    print("U,S,V shape: ", U.shape, S.shape, V.shape)

    print('explained variance', explained_variance)

    util_func = RiskMeasureUtil(util_func_key='mean_plus_sd', lambdaa=0.0)

    util_vals = util_func(Y)
    print('Y', Y)
    print('max over each y: ', torch.max(Y, dim=1))
    print('util_vals', util_vals)
