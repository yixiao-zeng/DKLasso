from itertools import islice

import gpytorch
import torch
from torch import nn
from torch.nn import functional as F
# from math import sqrt

from Prox import inplace_prox, hierar_prox


class FeatureExtractorLayer(nn.Module):
    def __init__(self, dims, res_dim, batch_norm, additive, dropout):
        """
        first dimension is input
        last dimension is output
        """
        assert len(dims) >= 3
        # self.additive = additive

        super(FeatureExtractorLayer, self).__init__()

        # self.dropout = nn.ModuleList([nn.Dropout(d) for d in dropout]) if dropout is not None else None
        self.dropout = nn.ModuleList([nn.Dropout(d) for d in dropout[:-1]]) if dropout is not None else None
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 2)]) if batch_norm is True else None
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
        )
        # self.layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
        if additive:
            self.skip = nn.Linear(dims[0], res_dim, bias=False)
            # self.batchnorm.append(nn.BatchNorm1d(round(sqrt(dims[0]))))
        else:
            self.skip = nn.Linear(dims[0], dims[-1], bias=False)
            # self.batchnorm.append(nn.BatchNorm1d(dims[-1]))


    def forward(self, inp):
        num_layers = len(self.layers)
        current_layer = inp
        skip_layer = self.skip(inp)
        for theta in range(num_layers):
            current_layer = self.layers[theta](current_layer)
            # if self.batchnorm is not None and theta != (num_layers-1):
            #     current_layer = self.batchnorm[theta](current_layer)
            # current_layer = F.relu(current_layer)
            # if self.dropout is not None:
            #     current_layer = self.dropout[theta](current_layer)
            if theta != (num_layers-1):
                if self.batchnorm is not None:
                    current_layer = self.batchnorm[theta](current_layer)
                current_layer = F.relu(current_layer)
                if self.dropout is not None:
                    current_layer = self.dropout[theta](current_layer)

        return skip_layer, current_layer
        # return self.batchnorm[-1](skip_layer), self.batchnorm[-2](current_layer)


    def prox(self, *, lambda_, lambda_bar=0, M=10):
        with torch.no_grad():
            inplace_prox(
                beta=self.skip,
                theta=self.layers[0],
                lambda_=lambda_,
                lambda_bar=lambda_bar,
                M=M,
            )


    def lambda_start(
        self,
        max_itr,
        M,
        lambda_bar=0,
        factor=2,
    ):
        """Estimate when the model will start to sparsify."""

        def is_sparse(lambda_):
            with torch.no_grad():
                beta = self.skip.weight.data
                theta = self.layers[0].weight.data

                for _ in range(max_itr):
                    new_beta, theta = hierar_prox(
                        beta,
                        theta,
                        lambda_=lambda_,
                        lambda_bar=lambda_bar,
                        M=M,
                    )
                    # if torch.norm((beta - new_beta), p=2, dim=0).sum() < 1e-5 * (torch.norm(beta, p=2, dim=0).sum()):
                    if torch.abs(beta - new_beta).max() < 1e-5:
                        break
                    beta = new_beta
                return (torch.norm(beta, p=2, dim=0) == 0).sum()

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start


    def l2_regularization(self):
        """
        L2 regulatization of the MLP without the first layer
        which is bounded by the skip connection
        """
        ans = 0
        for layer in islice(self.layers, 1, None):
            ans += (torch.norm(layer.weight.data, p=2)**2)
        return ans


    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()


    # def l2_regularization_skip(self):
    #     return torch.norm(self.skip.weight.data, p=2)**2


    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0


    def selected_count(self):
        return self.input_mask().sum().item()



class ApproximateGPLayer(gpytorch.models.ApproximateGP):
    """For the GP layer of the classification model"""
    def __init__(self, num_dims, kernel, grid_size, grid_bounds):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dims])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy, which places variational inducing points on a grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP, each task will be independent of one another
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution
            ), num_tasks=num_dims
        )
        super(ApproximateGPLayer, self).__init__(variational_strategy)

        # Each latent function has its own mean/kernel function
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_dims]))
        if kernel == "RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([num_dims])
                ), batch_shape=torch.Size([num_dims])
            )
        elif kernel == "Matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5, batch_shape=torch.Size([num_dims])
                ), batch_shape=torch.Size([num_dims])
            )
        else:
            raise SystemExit('Error: other custom kernels are not supportted yet.')
        # self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

