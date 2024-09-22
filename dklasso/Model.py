# import torch
import gpytorch
# import math
# from Utils import ScaleToBounds

from FunctionalLayers import ApproximateGPLayer

class ExactGP_Regressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, kernel, additive, grid_bounds):
        super(ExactGP_Regressor, self).__init__(train_x, train_y, likelihood)
        
        self.additive = additive
        self.feature_extractor = feature_extractor
        # These modules will scale the NN features so that they're nice values
        # self.scale_to_bounds = ScaleToBounds(scale_tanh=False, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])
        # self.scale_to_bounds_linear = ScaleToBounds(scale_tanh=False, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])

        # num_features = feature_extractor.layers[-1].out_features
        # num_features_skip = feature_extractor.skip.out_features

        # self.mean_module = gpytorch.means.LinearMean(input_size=num_features, bias=True)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel is None:
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_features)) + gpytorch.kernels.ConstantKernel()
            if additive:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel()
        # elif kernel == "Matern":
        #     self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)) + gpytorch.kernels.ConstantKernel()
        # elif kernel == "Periodic":
        #     self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ConstantKernel()
        # elif kernel == "Cosine":
        #     self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel()) + gpytorch.kernels.ConstantKernel()
        # elif kernel == "PiecewisePolynomial":
        #     self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=2)) + gpytorch.kernels.ConstantKernel()
        else:
            if isinstance(kernel, gpytorch.kernels.Kernel):
                self.covar_module = kernel
            else:
                raise SystemExit('Error: Non-GPyTorch kernels are not supported yet.')

        if additive:
            # self.mean_module_linear = gpytorch.means.LinearMean(input_size = num_features_skip, bias=False)
            # self.mean_module_linear = gpytorch.means.ZeroMean()
            self.covar_module_linear = gpytorch.kernels.LinearKernel()
        # else:
        #     self.scale_to_bounds_sum = ScaleToBounds(scale_tanh=False, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])


    def forward(self, x):
        # if self.additive:
        #     projected_x_linear, projected_x = self.feature_extractor(x)
        #     projected_x_linear = self.scale_to_bounds_linear(projected_x_linear)
        #     projected_x = self.scale_to_bounds(projected_x)

        #     mean = self.mean_module(projected_x)
        #     covar = self.covar_module(projected_x) + self.covar_module_linear(projected_x_linear)
        # else:
        #     projected_x = self.feature_extractor(x)
        #     projected_x = self.scale_to_bounds(projected_x)

        #     mean = self.mean_module(projected_x)
        #     covar = self.covar_module(projected_x)

        projected_x_linear, projected_x = self.feature_extractor(x)
        # projected_x_linear = self.scale_to_bounds_linear(projected_x_linear)  ## remove outliers in the linear projected inputs
        # projected_x = self.scale_to_bounds(projected_x)  ## remove outliers in the nonlinear transformed features

        if self.additive:
            mean = self.mean_module(projected_x)
            covar = self.covar_module(projected_x) + self.covar_module_linear(projected_x_linear)
        else:
            projected_x_sum = projected_x + projected_x_linear
            # projected_x_sum = self.scale_to_bounds_sum(projected_x_sum)

            mean = self.mean_module(projected_x_sum)
            covar = self.covar_module(projected_x_sum)
        
        return gpytorch.distributions.MultivariateNormal(mean, covar)


    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}



class ApproxGP_Regressor(gpytorch.models.ApproximateGP):
    def __init__(self, feature_extractor, inducing_points, kernel, additive, grid_bounds):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        # variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
        #     num_inducing_points=inducing_points.size(0)
        # )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True
        )
        super(ApproxGP_Regressor, self).__init__(variational_strategy)

        self.additive = additive
        self.feature_extractor = feature_extractor
        # self.scale_to_bounds = ScaleToBounds(scale_tanh=False, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])
        # self.scale_to_bounds_linear = ScaleToBounds(scale_tanh=False, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])

        # num_features = feature_extractor.layers[-1].out_features

        # self.mean_module = gpytorch.means.LinearMean(input_size=num_features, bias=True)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel is None:
            if additive:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel()
        else:
            if isinstance(kernel, gpytorch.kernels.Kernel):
                self.covar_module = kernel
            else:
                raise SystemExit('Error: Non-GPyTorch kernels are not supported yet.')

        if additive:
            # self.mean_module_linear = gpytorch.means.ZeroMean()
            self.covar_module_linear = gpytorch.kernels.LinearKernel()
        # else:
        #     self.scale_to_bounds_sum = ScaleToBounds(scale_tanh=False, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])


    def forward(self, x):
        projected_x_linear, projected_x = self.feature_extractor(x)
        # projected_x_linear = self.scale_to_bounds_linear(projected_x_linear)
        # projected_x = self.scale_to_bounds(projected_x)

        if self.additive:
            mean = self.mean_module(projected_x)
            covar = self.covar_module(projected_x) + self.covar_module_linear(projected_x_linear)
        else:
            projected_x_sum = projected_x + projected_x_linear
            # projected_x_sum = self.scale_to_bounds_sum(projected_x_sum)

            mean = self.mean_module(projected_x_sum)
            covar = self.covar_module(projected_x_sum)
        
        return gpytorch.distributions.MultivariateNormal(mean, covar)


    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}



class ApproxGP_Classifier(gpytorch.Module):
    def __init__(self, feature_extractor, kernel, grid_size, grid_bounds):
        super(ApproxGP_Classifier, self).__init__()

        self.feature_extractor = feature_extractor
        if grid_bounds == (-1., 1.):
            self.scale_to_bounds = ScaleToBounds(scale_tanh=True, scale_minmax=False, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])
        else:
            self.scale_to_bounds = ScaleToBounds(scale_tanh=False, scale_minmax=True, lower_bound=grid_bounds[0], upper_bound=grid_bounds[1])
        self.approxGP_layer = ApproximateGPLayer(num_dims=feature_extractor.layers[-1].out_features, kernel=kernel, grid_size=grid_size, grid_bounds=grid_bounds)


    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        # This next line makes it so that we learn a GP for each feature
        projected_x = projected_x.transpose(-1, -2).unsqueeze(-1)
        trained_dist = self.approxGP_layer(projected_x)

        return trained_dist


    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}

