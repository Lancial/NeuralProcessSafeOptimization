import torch
from random import randint
from neural_process import NeuralProcessImg
from torch import nn
from torch.distributions.kl import kl_divergence
from utils import *
from agents import *


        

class MovieNPTrainer():
    
    def __init__(self, device, neural_process, optimizer, init_context_num=5, max_opt_iteration=20, max_target_num=50, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.init_context_num = init_context_num
        self.max_opt_iteration = max_opt_iteration
        self.max_target_num = max_target_num
        self.steps = 0
        self.epoch_loss_history = []
        self.agents = EpsilonGreedyAgents()

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                

                x, y = data
                x, y, context_loc = sample_targets_and_save_contexts(x, y)
                if len(context_loc) == 0:
                    print('no safe initialization possible')
                    continue
                
                for step in range(min(self.max_opt_iteration, y.shape[1] - len(context_loc))):
                    
                    self.optimizer.zero_grad()
                    x_context, y_context, x_target, y_target = x[:, context_loc, :], y[:, context_loc, :], x, y
#                     print(context_loc, y_target[:, context_loc, :])
                    try:
                        p_y_pred, q_target, q_context = self.neural_process(x_context, y_context, x_target, y_target)
                    except ValueError:
                        print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)
                        print(context_loc)
                        
                    loss = self._loss(p_y_pred, y_target, q_target, q_context)
                    loss.backward()
                    self.optimizer.step()
                    
                    # call the agent to perform querying
                    mu = p_y_pred.loc.detach().numpy().squeeze(0)
                    sigma = p_y_pred.scale.detach().numpy().squeeze(0)
                    next_context = self.agents.get_next_query_point(mu, sigma, context_loc)
                    context_loc.append(next_context)
                    context_loc = sorted(context_loc)

                epoch_loss += loss.item()
                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl

class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = isinstance(self.neural_process, NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred, q_target, q_context = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                else:
                    x, y = data
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    p_y_pred, q_target, q_context = \
                        self.neural_process(x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
