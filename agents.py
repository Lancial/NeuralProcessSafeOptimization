import torch
import numpy as np
import random
from utils import *

class EpsilonGreedyAgents():
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
    
    def get_next_query_point(self, mu, sigma, context_loc):
        rest_loc = sorted([i for i in range(mu.shape[0]) if i not in context_loc])
        rest_mu = mu[rest_loc]
        rest_sigma = sigma[rest_loc]
        
        if np.random.rand(1) < self.epsilon:
            return random.choice(rest_loc)
        else: 
            return rest_loc[np.argmax(rest_mu)]
        

class NeuralProcessSafeOptAgents():
    
    def __init__(self, np_model, moive_user_features, ratings, beta=1.5, fmin=3.0, scaling=1):
        self.model = np_model
        self.x = moive_user_features
        self.y = ratings
        self.beta = beta
        self.fmin = fmin
        self.scaling = scaling
        self.context_loc = sorted(sample_save_contexts(self.x, self.y))
        if len(self.context_loc) == 0:
            print('no safe initialization exists')
        
        self.context_x = self.x[:, self.context_loc, :]
        self.context_y = self.y[:, self.context_loc, :]
        
        self.Q = np.empty((2, self.x.shape[1]), dtype=np.float)
        self.S = np.zeros(self.x.shape[1], dtype=np.bool)
        self.G = self.S.copy()
        self.M = self.S.copy()
    
    def reset(self):
        self.context_loc = sorted(sample_save_contexts(self.x, self.y))
        if len(self.context_loc) == 0:
            print('no safe initialization exists')
        
        self.context_x = self.x[:, self.context_loc, :]
        self.context_y = self.y[:, self.context_loc, :]
        
        self.Q = np.empty((2, self.x.shape[1]), dtype=np.float)
        self.S = np.zeros(self.x.shape[1], dtype=np.bool)
        self.G = self.S.copy()
        self.M = self.S.copy()
    
    def add_new_data_point(self, x, y, loc):
        if loc in self.context_loc:
            print('ignore duplicated context')
        else:
            print(x.shape, self.context_x.shape)
            self.context_loc.append(loc)
            self.context_x = torch.cat([self.context_x, x], dim=1)
            self.context_y = torch.cat([self.context_y, y], dim=1)
        

    
    def optimize(self):

        self.update_confidence_intervals()

        self.compute_sets()

        return self.get_new_query_point()
    
    def update_confidence_intervals(self):
        with torch.no_grad():
            p_y_pred = self.model(self.context_x, self.context_y, self.x)
        mean = p_y_pred.loc.numpy().squeeze()
        std_dev = p_y_pred.scale.numpy().squeeze()


        # Update confidence intervals
        self.Q[0, :] = mean - self.beta * std_dev
        self.Q[1, :] = mean + self.beta * std_dev
    
    def compute_safe_set(self):
        # Update safe set
        self.S[:] = self.Q[0, :] > self.fmin
  
    def compute_sets(self, full_sets=False):

        self.compute_safe_set()

        l, u = self.Q[0, :], self.Q[1, :]

        if not np.any(self.S):
            self.M[:] = False
            self.G[:] = False
            return

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[self.S] = u[self.S] >= np.max(l[self.S])
        max_var = np.max(u[self.M] - l[self.M]) / self.scaling
#         print(self.M, max_var)


        self.G[:] = False

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var or the threshold.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance
        if full_sets:
            s = self.S
        else:
            # skip points in M, they will already be evaluated
            
            s = np.logical_and(self.S, ~self.M)
            if not np.any(s):
                return
#             print(u[s])
#             s[s] = (np.max((u[s] - l[s]) / self.scaling, axis=1) >
#                     max_var)
            s[s] = ((u[s] - l[s]) / self.scaling) > max_var
#             s[s] = np.any(u[s, :] - l[s, :] > self.threshold * beta, axis=1)

            if not np.any(s):
                return

        def sort_generator(array):
            """Return the sorted array, largest element first."""
            return array.argsort()[::-1]

        # set of safe expanders
        G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

        if not full_sets:
            # Sort, element with largest variance first
            sort_index = sort_generator(u[s] - l[s])
        else:
            # Sort index is just an enumeration of all safe states
            sort_index = range(len(G_safe))

        for index in sort_index:
                
            temp_context_x = torch.cat([self.context_x, self.x[:, s, :][:, index, :].unsqueeze(0)], dim=1)
            optimistic_y = torch.from_numpy(np.array([u[s][index]])[np.newaxis, ..., np.newaxis])
            assert optimistic_y.shape == (1, 1, 1)
            temp_context_y = torch.cat([self.context_y, optimistic_y], dim=1)
            with torch.no_grad():
                print(type(temp_context_x), type(temp_context_y), type(self.x[:, ~self.S, :]))
                print(temp_context_x.shape, temp_context_y.shape, self.x[:, ~self.S, :].shape)
                p_y_pred = self.model(temp_context_x.float(), temp_context_y.float(), self.x[:, ~self.S, :].float())
                    
            # Prediction of previously unsafe points based on that
            mean2, var2 = p_y_pred.loc.numpy(), p_y_pred.scale.numpy()

            mean2 = mean2.squeeze()
            var2 = var2.squeeze()
            l2 = mean2 -self.beta * np.sqrt(var2)

            # If any unsafe lower bound is suddenly above fmin then
            # the point is an expander
            G_safe[index] = np.any(l2 >= self.fmin)

            # Since we sorted by uncertainty and only the most
            # uncertain element gets picked by SafeOpt anyways, we can
            # stop after we found the first one
            if G_safe[index] and not full_sets:
                break

        # Update safe set (if full_sets is False this is at most one point
        self.G[s] = G_safe
#         print(np.any(G_Safe))
        if np.any(self.G):
            print("GGGGG", self.G)
    
    def get_new_query_point(self):
        if not np.any(self.S):
            raise EnvironmentError('There are no safe points to evaluate.')
            
        l = self.Q[0, :]
        u = self.Q[1, :]

        MG = np.logical_or(self.M, self.G)
#         print(MG.shape, u.shape)
#         value = np.max((u[MG] - l[MG]) / self.scaling, axis=1)
        value = (u[MG] - l[MG]) / self.scaling
        x = self.x[:, MG, :][:, np.argmax(value), :]
        
        positive_count = -1
        for i in range(MG.shape[0]):
            if MG[i]:
                positive_count += 1
            if positive_count == np.argmax(value):
                loc = i
                break
            
            
        return x, loc
