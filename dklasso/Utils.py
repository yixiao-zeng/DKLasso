import torch
# import gpytorch
# import math
# import numpy
# import scipy.stats

# def initialize_state_from_data(train_x, train_y, test_x, test_y, model, gamma, n_epochs=500):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = gamma)
#     criterion = torch.nn.MSELoss(reduction="mean")

#     best_test_loss = float("inf")
#     epochs_since_best = 0
#     for epoch in range(n_epochs):
#         model.train()
#         optimizer.zero_grad()
#         loss = criterion(model(train_x), train_y.view(-1,1))
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             test_loss = criterion(model(test_x), test_y.view(-1,1)).item()

#         if test_loss < 0.99 * best_test_loss:
#             best_test_loss = test_loss
#             epochs_since_best = 0
#         else:
#             epochs_since_best += 1

#         if epochs_since_best == 20:
#             break

#     return {k: v.detach().clone().cpu() for k, v in model.feature_extractor.state_dict().items()}

def initialize_optimizer(model, likelihood, method, model_type, additive, lr_init, lr_path, gamma, batch_norm):
    feature_extractor_layers_bias = []
    feature_extractor_layers_weight = []
    # start from the second layer, and exclude the last layer
    for param in model.feature_extractor.layers[1:-1].parameters():
        # exclude bias from weight decay
        if param.ndim == 1:
            feature_extractor_layers_bias.append(param)
        else:
            feature_extractor_layers_weight.append(param)

    if method == "Regressor":
        if additive:
            if model_type == "Exact":
                optim_init = torch.optim.Adam([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.covar_module_linear.parameters()},
                    # {'params': model.mean_module_linear.parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_init)

                optim_path = torch.optim.SGD([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.covar_module_linear.parameters()},
                    # {'params': model.mean_module_linear.parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_path, momentum=0.9, nesterov=True)

                if batch_norm:
                    optim_init.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
                    optim_path.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
            
            elif model_type == "Approximate":
                optim_init = torch.optim.Adam([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.covar_module_linear.parameters()},
                    # {'params': model.mean_module_linear.parameters()},
                    {'params': model.variational_parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_init)

                optim_path = torch.optim.SGD([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.covar_module_linear.parameters()},
                    # {'params': model.mean_module_linear.parameters()},
                    {'params': model.variational_parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_path, momentum=0.9, nesterov=True)

                if batch_norm:
                    optim_init.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
                    optim_path.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
        
        else:
            if model_type == "Exact":
                optim_init = torch.optim.Adam([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_init)

                optim_path = torch.optim.SGD([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_path, momentum=0.9, nesterov=True)

                if batch_norm:
                    optim_init.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
                    optim_path.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
            
            elif model_type == "Approximate":
                # variational_ngd_optim = gpytorch.optim.NGD(
                #     model.variational_parameters(), num_data=num_data, lr=0.1
                # )
                optim_init = torch.optim.Adam([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.variational_parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_init)

                optim_path = torch.optim.SGD([
                    # {'params': model.feature_extractor.parameters()},
                    {'params': model.feature_extractor.skip.parameters()},
                    {'params': model.feature_extractor.layers[0].parameters()},
                    {'params': model.feature_extractor.layers[-1].parameters()},
                    {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
                    {'params': feature_extractor_layers_bias},
                    # {'params': model.feature_extractor.batchnorm.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.variational_parameters()},
                    {'params': likelihood.parameters()},
                ], lr=lr_path, momentum=0.9, nesterov=True)

                if batch_norm:
                    optim_init.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
                    optim_path.add_param_group({'params': model.feature_extractor.batchnorm.parameters()})
                
            # lr_scheduler_init = CosineScheduler((offset_init-5), warmup_steps=round((offset_init-5)/4), warmup_begin_lr=lr_init, base_lr=lr_init*100, final_lr=lr_init)
            # lr_scheduler_path = CosineScheduler((offset_path-5), warmup_steps=round((offset_path-5)/4), warmup_begin_lr=lr_path, base_lr=lr_path*10, final_lr=lr_path)


    elif method == "Classifier":
        optim_init = torch.optim.Adam([
            # {'params': model.feature_extractor.parameters()},
            {'params': model.feature_extractor.skip.parameters()},
            {'params': model.feature_extractor.layers[0].parameters()},
            {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
            {'params': feature_extractor_layers_bias},
            {'params': model.approxGP_layer.variational_parameters()},
            {'params': model.approxGP_layer.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=lr_init)

        optim_path = torch.optim.SGD([
            # {'params': model.feature_extractor.parameters()},
            {'params': model.feature_extractor.skip.parameters()},
            {'params': model.feature_extractor.layers[0].parameters()},
            {'params': feature_extractor_layers_weight, 'weight_decay': gamma},
            {'params': feature_extractor_layers_bias},
            {'params': model.approxGP_layer.variational_parameters()},
            {'params': model.approxGP_layer.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=lr_path, momentum=0.9, nesterov=False)

    return(optim_init, optim_path)



class ScaleToBounds(torch.nn.Module):
    """
    Scale the input data so that it lies in between the lower and upper bounds.

    In training (`self.train()`), this module adjusts the scaling factor to the minibatch of data.
    During evaluation (`self.eval()`), this module uses the scaling factor from the previous minibatch of data.

    :param float lower_bound: lower bound of scaled data
    :param float upper_bound: upper bound of scaled data

    Example:
        >>> train_x = torch.randn(10, 5)
        >>> module = ScaleToBounds(lower_bound=-1., upper_bound=1.)
        >>>
        >>> module.train()
        >>> scaled_train_x = module(train_x)  # Data should be between -0.95 and 0.95
        >>>
        >>> module.eval()
        >>> test_x = torch.randn(10, 5)
        >>> scaled_test_x = module(test_x)  # Scaling is based on train_x
    """

    def __init__(self, scale_tanh, scale_minmax, lower_bound, upper_bound):
        super().__init__()
        self.scale_tanh = scale_tanh
        self.scale_minmax = scale_minmax
        if scale_tanh:
            self.scale_layer = torch.nn.Tanh()
        if scale_minmax:
            self.lower_bound = float(lower_bound)
            self.upper_bound = float(upper_bound)
        self.register_buffer("min_val", torch.tensor(lower_bound))
        self.register_buffer("max_val", torch.tensor(upper_bound))

    def forward(self, x):
        if self.scale_tanh:
            x = self.scale_layer(x)
        else:
            if self.training:
                with torch.no_grad():
                    Q3 = torch.quantile(x, q=0.75, dim=0, keepdim=True)
                    Q1 = torch.quantile(x, q=0.25, dim=0, keepdim=True)
                    IQR = Q3 - Q1
                    clamp_min = Q1 - 3.0*IQR
                    clamp_max = Q3 + 3.0*IQR
                # Clamp extreme values
                x = x.clamp(clamp_min, clamp_max)

                with torch.no_grad():
                    min_val = x.min(dim=0, keepdim=True).values
                    max_val = x.max(dim=0, keepdim=True).values
                self.min_val.data = min_val
                self.max_val.data = max_val
            else:
                min_val = self.min_val
                max_val = self.max_val
                # Clamp extreme values
                x = x.clamp(min_val, max_val)
        
            if self.scale_minmax:
                diff = max_val - min_val
                x = (x - min_val) * (0.95 * (self.upper_bound - self.lower_bound) / diff) + 0.95 * self.lower_bound
        return x



# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0,
#     warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps

#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) \
#                        * float(epoch) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase

#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (
#                 self.base_lr_orig - self.final_lr) * (1 + math.cos(
#                 math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
#         return self.base_lr



# def initialize_state_from_data(train_x, train_y, test_x, test_y, model, gamma, gamma_skip):
#     if model.additive:
#         optimizer = torch.optim.Adam([
#             {'params': model.feature_extractor.parameters()},
#             {'params': model.covar_module.parameters()},
#             {'params': model.mean_module.parameters()},
#             {'params': model.covar_module_linear.parameters()},
#             {'params': model.mean_module_linear.parameters()},
#             {'params': model.likelihood.parameters()},
#         ], lr=1e-3, weight_decay=0)
#     else:
#         optimizer = torch.optim.Adam([
#             {'params': model.feature_extractor.parameters()},
#             {'params': model.covar_module.parameters()},
#             {'params': model.mean_module.parameters()},
#             {'params': model.likelihood.parameters()},
#         ], lr=1e-3, weight_decay=0)

#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)

#     epochs_since_best = 0
#     best_test_objective = float("inf")
#     for epoch in range(200):
#         model.train()
#         model.likelihood.train()
#         with gpytorch.settings.num_likelihood_samples(32):
#             optimizer.zero_grad()
#             output = model(train_x)
#             loss = -mll(output, train_y) + gamma * model.feature_extractor.l2_regularization() + gamma_skip * model.feature_extractor.l2_regularization_skip()
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         model.likelihood.eval()
#         with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(32):
#             pred_y = model.likelihood(model(test_x))
#             test_objective = torch.mean((pred_y.mean-test_y)**2).item() + gamma * model.feature_extractor.l2_regularization().item() + gamma_skip * model.feature_extractor.l2_regularization_skip().item()
#         if test_objective < best_test_objective:
#             best_test_objective = test_objective
#             epochs_since_best = 0
#         else:
#             epochs_since_best += 1
#         if epochs_since_best == 20:
#             break

#     return {k: v.detach().clone().cpu() for k, v in model.feature_extractor.state_dict().items()}

