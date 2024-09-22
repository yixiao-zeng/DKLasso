import torch
import gpytorch
#import itertools
import warnings
from typing import List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from itertools import islice
from math import sqrt

from sklearn.base import BaseEstimator

from FunctionalLayers import FeatureExtractorLayer
from Model import ExactGP_Regressor, ApproxGP_Regressor, ApproxGP_Classifier
from Utils import initialize_optimizer


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    likelihood_state_dict: dict
    train_loss: float
    test_loss: float
    test_acc: float
    l1_regularization_skip: float
    l2_regularization: float
    # l2_regularization_skip: float
    selected: torch.BoolTensor
    n_iters: int

    def log(item):
        if item.test_acc is None:
            print(
                f"{item.n_iters} epochs, "
                f"train_loss (NMLL) "
                f"{item.train_loss:.3e}, "
                f"test_loss "
                f"{item.test_loss:.3e}, "
                f"l2_reg {item.l2_regularization:.3e}, "
                # f"l2_reg_skip {item.l2_regularization_skip:.3e}, "
                f"l1_reg_skip {item.l1_regularization_skip:.3e}"
            )
        else:
            print(
                f"{item.n_iters} epochs, "
                f"train_loss (NMLL) "
                f"{item.train_loss:.3e}, "
                f"test_loss (CE) "
                f"{item.test_loss:.3e}, "
                f"test_accuracy "
                f"{item.test_acc:.3f}, "
                f"l2_reg {item.l2_regularization:.3e}, "
                # f"l2_reg_skip {item.l2_regularization_skip:.3e}, "
                f"l1_reg_skip {item.l1_regularization_skip:.3e}"
            )


class DKLasso(BaseEstimator):
    def __init__(
        self,
        *,
        method="Regressor",
        model_type="Exact",
        kernel=None,
        additive=True,
        hidden_dims=(100,2),
        res_dim=None,
        batch_norm=None,
        lambda_start=None,
        lambda_max=float("inf"),
        path_multiplier=(1.02, 1.02),  # before and after the model begins to sparsify
        M=10.0,
        gamma=1.0e-4,  # weight decay of the MLP without the first layer
        # gamma_skip=0.0,
        dropout=(0.2,),
        lr=(1e-3, 1e-3),
        n_epochs=(1000, 100),
        patience=(100, 2),
        offset=(1, 1),
        grid_size=100,
        grid_bounds=(-1., 1.),
        batch_size=None,
        val_size=0.15,
        device=None,
        verbose=1,
        random_state=None,
        torch_seed=None
    ):
        self.method = method
        self.kernel = kernel
        self.hidden_dims = hidden_dims
        self.res_dim = res_dim
        self.lambda_start = lambda_start
        self.lambda_max = lambda_max
        self.path_multiplier = path_multiplier
        self.M = M
        self.gamma = gamma
        # self.gamma_skip = gamma_skip
        self.lr_init, self.lr_path = lr
        self.n_epochs_init, self.n_epochs_path = n_epochs
        self.patience_init, self.patience_path = patience
        self.offset_init, self.offset_path = offset
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        self.batch_size = batch_size
        self.val_size = val_size
        self.verbose = verbose
        self.random_state = random_state
        self.torch_seed = torch_seed

        if dropout is not None:
            if len(dropout) == 1:
                # self.dropout = dropout * (len(hidden_dims)-1)
                self.dropout = dropout * len(hidden_dims)
            # elif len(dropout) == (len(hidden_dims)-1):
            elif len(dropout) == len(hidden_dims):
                self.dropout = dropout
            else:
                raise SystemExit('Error: `dropout` should be a tuple of length 1 or len(hidden_dims).')
        else:
            self.dropout = None

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if method == "Regressor":
            self.model_type = model_type
            self.additive = additive
        elif method == "Classifier":
            self.model_type = "Approximate"
            self.additive = False
        else:
            raise SystemExit('Error: `method` has to be either "Regressor" or "Classifier".')
        
        if batch_norm is None:
            self.batch_norm = False if self.model_type == "Exact" else True
        else:
            self.batch_norm = batch_norm


    def _cast_input(self, x, y=None):
        x = torch.FloatTensor(x).to(self.device)
        if y is None:
            return x

        def _convert_y(y):
            if self.method == "Regressor":
                y = torch.FloatTensor(y).to(self.device)
                return y
            elif self.method == "Classifier":
                if isinstance(y, torch.Tensor):
                    y = y.to(torch.long).to(self.device)
                    return y
                else:
                    y = torch.LongTensor(y).to(self.device)
                    return y

        y = _convert_y(y)
        return x, y


    def _init_model(self, train_x=None, train_y=None):
        inp_dim = train_x.size(-1)
        dims = (inp_dim,) + self.hidden_dims
        if self.res_dim is None:
            self.res_dim = round(sqrt(inp_dim))

        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)

        if self.method == "Regressor":
            feature_extractor = FeatureExtractorLayer(dims=dims, res_dim=self.res_dim, batch_norm=self.batch_norm, additive=self.additive, dropout=self.dropout).to(self.device)
            # if self.kernel == "SM":
            #     init_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            #     # feature_extractor_nn = FeatureExtractorFullNet(feature_extractor=feature_extractor)
            #     init_model = ExactGP_Model(train_x=train_x, train_y=train_y, likelihood=init_likelihood, feature_extractor=feature_extractor, kernel="RBF", additive=self.additive, grid_bounds=self.grid_bounds)
            #     # pretrained_state = initialize_from_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, model=feature_extractor_nn, gamma=self.gamma, n_epochs=500)
            #     init_state = initialize_state_from_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, model=init_model, gamma=self.gamma, gamma_skip=self.gamma_skip)
            #     feature_extractor.load_state_dict(init_state)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            if self.model_type == "Exact":
                self.model = ExactGP_Regressor(train_x=train_x, train_y=train_y, likelihood=self.likelihood, feature_extractor=feature_extractor, kernel=self.kernel, additive=self.additive, grid_bounds=self.grid_bounds).to(self.device)
                self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=self.likelihood, model=self.model)
            elif self.model_type == "Approximate":
                inducing_points = train_x[list(range(0, train_y.numel(), round(train_y.numel()/self.grid_size))), :]
                self.model = ApproxGP_Regressor(feature_extractor=feature_extractor, inducing_points=inducing_points, kernel=self.kernel, additive=self.additive, grid_bounds=self.grid_bounds).to(self.device)
                self.mll = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood, model=self.model, num_data=train_y.numel())
        elif self.method == "Classifier":
            feature_extractor = FeatureExtractorLayer(dims=dims, res_dim=self.res_dim, batch_norm=self.batch_norm, additive=False, dropout=self.dropout).to(self.device)
            self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=dims[-1], num_classes=len(torch.unique(train_y))).to(self.device)
            self.model = ApproxGP_Classifier(feature_extractor=feature_extractor, kernel=self.kernel, grid_size=self.grid_size, grid_bounds=self.grid_bounds).to(self.device)
            self.mll = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood, model=self.model.approxGP_layer, num_data=train_y.numel())


    def _train(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        n_epochs,
        lambda_,
        gamma,
        # gamma_skip,
        lr,
        optimizer,
        offset,
        patience,
        return_state_dict=True
    ) -> HistoryItem:

        if self.method == "Regressor":
            def _test_eval():
                self.model.eval()
                self.likelihood.eval()
                # with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.skip_posterior_variances(True), gpytorch.settings.max_root_decomposition_size(100):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    # pred_dist = self.likelihood(self.model(test_x))  ## distribution of y*
                    pred_dist = self.model(test_x)  ## distribution of f*
                    # test_loss = torch.square(pred_dist.mean-test_y).mean()
                    # test_loss = -torch.exp(-gpytorch.metrics.mean_standardized_log_loss(pred_dist, test_y=test_y))  ## negative likelihood, lower is better
                    test_loss = gpytorch.metrics.mean_standardized_log_loss(pred_dist, test_y=test_y)
                    test_objective = test_loss + lambda_ * self.model.feature_extractor.l1_regularization_skip()
                return test_loss, test_objective
            
            
            if self.model_type == "Exact":
                # tol = 0.99 if optimizer == "optim_path" else 1.0
                epochs_since_best = 0
                for epoch in range(n_epochs):
                    self.model.train()
                    self.likelihood.train()

                    n_iters = epoch + 1
                    if optimizer == "optim_path":
                        # with gpytorch.settings.use_toeplitz(False):
                        self.optim_path.zero_grad()
                        output = self.model(train_x)
                        # loss = -self.mll(output, train_y) + gamma * self.model.feature_extractor.l2_regularization() + gamma_skip * self.model.feature_extractor.l2_regularization_skip()
                        loss = -self.mll(output, train_y)
                        loss.backward()
                        self.optim_path.step()
                        # for param_group in self.optim_path.param_groups:
                        #     param_group['lr'] = self.lr_scheduler_path(epoch)
                        self.model.feature_extractor.prox(lambda_=lambda_*lr, M=self.M)
                    else:
                        # with gpytorch.settings.use_toeplitz(False):
                        self.optim_init.zero_grad()
                        output = self.model(train_x)
                        # loss = -self.mll(output, train_y) + gamma * self.model.feature_extractor.l2_regularization() + gamma_skip * self.model.feature_extractor.l2_regularization_skip()
                        loss = -self.mll(output, train_y)
                        loss.backward()
                        self.optim_init.step()
                        # for param_group in self.optim_init.param_groups:
                        #     param_group['lr'] = self.lr_scheduler_init(epoch)

                    if self.model.feature_extractor.selected_count() > 0:
                        if n_iters > offset and ((n_iters - offset) % 2 == 0):
                            test_loss, test_objective = _test_eval()
                            # if test_objective < tol * best_test_objective:
                            if test_objective < best_test_objective:
                            # if test_loss < best_test_loss:
                            # if torch.abs(test_objective - prev_test_objective) > (1-tol) * torch.abs(prev_test_objective):
                                best_test_loss, best_test_objective = test_loss, test_objective
                                # prev_test_loss, prev_test_objective = test_loss, test_objective
                                epochs_since_best = 0
                            else:
                                epochs_since_best += 2
                                if epochs_since_best >= patience:
                                    break
                                # if (n_iters - offset) == 4 and optimizer == "optim_path":
                                #     if epochs_since_best == 2:
                                #         patience = min(4, patience)
                                #     elif epochs_since_best == 4:
                                #         break
                                if (n_iters - offset) == 2 and epochs_since_best == 2:
                                    if optimizer == "optim_path":
                                        break
                                # if test_objective < best_test_objective:
                                #     best_test_loss, best_test_objective = test_loss, test_objective
                                # prev_test_loss, prev_test_objective = test_loss, test_objective
                        elif n_iters == offset:
                            best_test_loss, best_test_objective = _test_eval()
                            # prev_test_loss, prev_test_objective = _test_eval()
                        else:
                            continue
                    else:
                        test_loss, test_objective = _test_eval()
                        break

            elif self.model_type == "Approximate":
                n_train = train_y.numel()
                randperm = torch.arange if self.batch_size == n_train else torch.randperm
                num_batches = n_train // self.batch_size  # omit batches that are not full
                # tol = 0.99 if optimizer == "optim_path" else 1.0
                
                epochs_since_best = 0
                for epoch in range(n_epochs):
                    self.model.train()
                    self.likelihood.train()

                    n_iters = epoch + 1
                    if self.torch_seed is not None:
                        torch.manual_seed(self.torch_seed + epoch**2 + round((lambda_/10)**2))
                    indices = randperm(n_train)
                    loss = torch.FloatTensor([0]).to(self.device)

                    for i in range(num_batches):
                        batch = indices[i*self.batch_size : (i+1)*self.batch_size]

                        if optimizer == "optim_path":
                            # with gpytorch.settings.use_toeplitz(False):
                            self.optim_path.zero_grad()
                            output = self.model(train_x[batch])
                            # batch_loss = -self.mll(output, train_y[batch]) + gamma * self.model.feature_extractor.l2_regularization() + gamma_skip * self.model.feature_extractor.l2_regularization_skip()
                            batch_loss = -self.mll(output, train_y[batch])
                            batch_loss.backward()
                            self.optim_path.step()
                            self.model.feature_extractor.prox(lambda_=lambda_*lr, M=self.M)
                        else:
                            # with gpytorch.settings.use_toeplitz(False):
                            self.optim_init.zero_grad()
                            output = self.model(train_x[batch])
                            # batch_loss = -self.mll(output, train_y[batch]) + gamma * self.model.feature_extractor.l2_regularization() + gamma_skip * self.model.feature_extractor.l2_regularization_skip()
                            batch_loss = -self.mll(output, train_y[batch])
                            batch_loss.backward()
                            self.optim_init.step()
                        
                        with torch.no_grad():
                            loss += batch_loss * 1.0 / num_batches

                    # if optimizer == "optim_path":
                    #     for param_group in self.optim_path.param_groups:
                    #         param_group['lr'] = self.lr_scheduler_path(epoch)
                    # else:
                    #     for param_group in self.optim_init.param_groups:
                    #         param_group['lr'] = self.lr_scheduler_init(epoch)

                    if self.model.feature_extractor.selected_count() > 0:
                        if n_iters > offset and ((n_iters - offset) % 2 == 0):
                            test_loss, test_objective = _test_eval()
                            # if test_objective < tol * best_test_objective:
                            if test_objective < best_test_objective:
                                best_test_loss, best_test_objective = test_loss, test_objective
                                epochs_since_best = 0
                            else:
                                epochs_since_best += 2
                                if epochs_since_best >= patience:
                                    break
                                if (n_iters - offset) == 2 and epochs_since_best == 2:
                                    if optimizer == "optim_path":
                                        break
                                # if test_objective < best_test_objective:
                                #     best_test_loss, best_test_objective = test_loss, test_objective
                        elif n_iters == offset:
                            best_test_loss, best_test_objective = _test_eval()
                        else:
                            continue
                    else:
                        test_loss, test_objective = _test_eval()
                        break
                        
            with torch.no_grad():
                l1_reg_skip = self.model.feature_extractor.l1_regularization_skip().item()
                l2_reg = self.model.feature_extractor.l2_regularization().item()
                # l2_reg_skip = self.model.feature_extractor.l2_regularization_skip().item()

            return HistoryItem(
                lambda_=lambda_,
                state_dict=self.model.cpu_state_dict() if return_state_dict else None,
                likelihood_state_dict=None,
                train_loss=loss.item(),
                #train_objective=(loss.item() + lambda_ * l1_reg) if self.model_type == "Exact" else (loss + lambda_ * l1_reg),
                test_loss=test_loss.item(),
                #test_objective=test_objective.item(),
                test_acc=None,
                l1_regularization_skip=l1_reg_skip,
                l2_regularization=l2_reg,
                # l2_regularization_skip=l2_reg_skip,
                selected=self.model.feature_extractor.input_mask().cpu(),
                n_iters=n_iters
            )

        elif self.method == "Classifier":
            def cross_entropy(probs, y):
                return -torch.log(probs[range(len(probs)), y])

            def _test_eval():
                self.model.eval()
                self.likelihood.eval()

                with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.num_likelihood_samples(64):
                    output = self.likelihood(self.model(test_x))
                    pred_probs = output.probs.mean(0)
                    test_loss = torch.mean(cross_entropy(pred_probs, test_y))
                    test_objective = test_loss + lambda_ * self.model.feature_extractor.l1_regularization_skip()

                    pred_class = pred_probs.argmax(-1)
                    correct = pred_class.eq(test_y.view_as(pred_class)).sum()
                    test_acc = correct / test_y.numel()

                return test_loss, test_objective, test_acc


            n_train = train_y.numel()
            randperm = torch.arange if self.batch_size == n_train else torch.randperm
            num_batches = n_train // self.batch_size
            tol = 0.98 if optimizer == "optim_path" else 1.0

            epochs_since_best = 0
            for epoch in range(n_epochs):
                self.model.train()
                self.likelihood.train()

                n_iters = epoch + 1
                if self.torch_seed is not None:
                    torch.manual_seed(self.torch_seed + epoch**2 + round(lambda_))
                indices = randperm(n_train)
                loss = torch.FloatTensor([0]).to(self.device)
                for i in range(num_batches):
                    # omit batches that are not full
                    batch = indices[i*self.batch_size : (i+1)*self.batch_size]

                    if optimizer == "optim_path":
                        with gpytorch.settings.num_likelihood_samples(64), gpytorch.settings.use_toeplitz(False):
                            self.optim_path.zero_grad()
                            output = self.model(train_x[batch])
                            # batch_loss = -self.mll(output, train_y[batch]) + gamma * self.model.feature_extractor.l2_regularization() + gamma_skip * self.model.feature_extractor.l2_regularization_skip()
                            batch_loss = -self.mll(output, train_y[batch])
                            batch_loss.backward()
                            self.optim_path.step()
                    else:
                        with gpytorch.settings.num_likelihood_samples(64), gpytorch.settings.use_toeplitz(False):
                            self.optim_init.zero_grad()
                            output = self.model(train_x[batch])
                            # batch_loss = -self.mll(output, train_y[batch]) + gamma * self.model.feature_extractor.l2_regularization() + gamma_skip * self.model.feature_extractor.l2_regularization_skip()
                            batch_loss = -self.mll(output, train_y[batch])
                            batch_loss.backward()
                            self.optim_init.step()
                    self.model.feature_extractor.prox(lambda_=lambda_*lr, M=self.M)
                    with torch.no_grad():
                        loss += batch_loss * 1.0 / num_batches
                
                if self.model.feature_extractor.selected_count() > 0:
                    if n_iters > offset and ((n_iters - offset) % 4 == 0):
                        test_loss, test_objective, test_acc = _test_eval()
                        if test_objective < tol * best_test_objective:
                            best_test_loss, best_test_objective, best_test_acc = test_loss, test_objective, test_acc
                            epochs_since_best = 0
                        else:
                            epochs_since_best += 4
                            if epochs_since_best >= patience:
                                break
                            if (n_iters - offset) == 8:
                                if epochs_since_best == 4 and optimizer == "optim_path":
                                    patience = min(10, patience)
                                elif epochs_since_best == 8 and optimizer == "optim_path":
                                    break
                            if test_objective < best_test_objective:
                                best_test_loss, best_test_objective, best_test_acc = test_loss, test_objective, test_acc
                    elif n_iters == offset:
                        best_test_loss, best_test_objective, best_test_acc = _test_eval()
                    else:
                        continue
                else:
                    test_loss, test_objective, test_acc = _test_eval()
                    break

            with torch.no_grad():
                l1_reg_skip = self.model.feature_extractor.l1_regularization_skip().item()
                l2_reg = self.model.feature_extractor.l2_regularization().item()
                # l2_reg_skip = self.model.feature_extractor.l2_regularization_skip().item()

            return HistoryItem(
                lambda_=lambda_,
                state_dict=self.model.cpu_state_dict() if return_state_dict else None,
                likelihood_state_dict={k: v.detach().clone().cpu() for k, v in self.likelihood.state_dict().items()} if return_state_dict else None,
                train_loss=loss.item(),
                test_loss=test_loss.item(),
                test_acc=test_acc.item(),
                l1_regularization_skip=l1_reg_skip,
                l2_regularization=l2_reg,
                # l2_regularization_skip=l2_reg_skip,
                selected=self.model.feature_extractor.input_mask().cpu(),
                n_iters=n_iters
            )


    def train_path(
        self,
        x,
        y,
        test_x=None,
        test_y=None,
    ) -> List[HistoryItem]:

        assert (test_x is None) == (test_y is None), "Must specify both or none of test_x and test_y"
        # assert len(y.shape) == 1, "y (and test_y) must be a vector (not torch.Size([n, 1]))"
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        if len(y.shape) == 2:
            y = y.squeeze(-1)
    
        sample_test = (self.val_size != 0) and (test_x is None)
        if sample_test:
            train_x, test_x, train_y, test_y = train_test_split(
                x, y, test_size=self.val_size, random_state=self.random_state
            )
        elif test_x is None:
            train_x, train_y = test_x, test_y = x, y
        else:
            train_x, train_y = x, y
            if len(test_x.shape) == 1:
                test_x = test_x.unsqueeze(-1)

        train_x, train_y = self._cast_input(x=train_x, y=train_y)
        test_x, test_y = self._cast_input(x=test_x, y=test_y)

        if self.model_type == "Approximate":
            if self.batch_size is None:
                self.batch_size = train_y.numel()
            self.batch_size = min(self.batch_size, train_y.numel())


        HistoryList: List[HistoryItem] = []

        # initialize model
        self._init_model(train_x=train_x, train_y=train_y)

        self.optim_init, self.optim_path = initialize_optimizer(
            model=self.model,
            likelihood=self.likelihood,
            method=self.method,
            model_type=self.model_type,
            additive=self.additive,
            lr_init=self.lr_init,
            lr_path=self.lr_path,
            gamma=self.gamma,
            batch_norm=self.batch_norm
        )

        HistoryList.append(
            self._train(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                n_epochs=self.n_epochs_init,
                lambda_=0,
                gamma=self.gamma,
                # gamma_skip=self.gamma_skip,
                lr=self.lr_init,
                optimizer="optim_init",
                #optimizer=self.optim_init,
                offset=self.offset_init,
                patience=self.patience_init,
                return_state_dict=True
            )
        )

        print("\nInitialized DKL model")
        HistoryList[-1].log()
        
        ## TO DO: user-specified lambda sequence ##
        if self.lambda_start is None:
            self.lambda_start = (
                self.model.feature_extractor.lambda_start(max_itr=10000, M=self.M) / self.lr_path / 5.0
            )
        
        is_dense = True
        lambda_current = self.lambda_start / self.path_multiplier[0]
        n_lambda = 0
        while self.model.feature_extractor.selected_count() > 0 and lambda_current <= self.lambda_max:
        #for lambda_current in itertools.chain([lambda_start], lambda_seq):
            #if self.model.feature_extractor.selected_count() == 0:
            #    break
            lambda_current = lambda_current * self.path_multiplier[0] if is_dense else lambda_current * self.path_multiplier[1]

            latest = self._train(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                n_epochs=self.n_epochs_path,
                lambda_=lambda_current,
                gamma=self.gamma,
                # gamma_skip=self.gamma_skip,
                lr=self.lr_path,
                optimizer="optim_path",
                #optimizer=self.optim_path,
                offset=self.offset_path,
                patience=self.patience_path,
                return_state_dict=True
            )
            HistoryList.append(latest)
            n_lambda += 1

            if is_dense and self.model.feature_extractor.selected_count() < train_x.shape[-1]:
                is_dense = False
                if lambda_current / self.lambda_start < 1.5:
                    warnings.warn(
                        f"lambda_start = {self.lambda_start:.3f} "
                        "might be too large.\n"
                        f"Features start to disappear at {lambda_current=:.3f}."
                    )
       
            if self.verbose > 0:
                print(
                    f"\n{n_lambda}th lambda = "
                    f"{lambda_current:.3e}, "
                    f"selected {self.model.feature_extractor.selected_count()} features"
                )
                latest.log()
                # if self.method == "Classifier":
                #     print('Test set accuracy: {}%'.format(100. * HistoryList[-1].test_acc))

        self.feature_importances = self._compute_feature_importances(HistoryList)

        return HistoryList


    @staticmethod
    def _compute_feature_importances(path: List[HistoryItem]):
        """When does each feature disappear on the path?
        Parameters
        ----------
        path : List[HistoryItem]
        Returns
        -------
            feature_importances_
        """

        current = path[0].selected.clone()
        imp = torch.full(current.shape, float("inf"))
        for save in islice(path, 1, None):
            lambda_ = save.lambda_
            diff = current & ~save.selected
            imp[diff.nonzero().flatten()] = lambda_
            current &= save.selected
        return imp

