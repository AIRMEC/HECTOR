# Used from https://github.com/mahmoodlab/MCAT

import torch
import numpy as np

# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = 0 if T_cont \in (-inf, 0), Y = 1 if T_cont \in [0, a_1),  Y = 2 if T_cont in [a_1, a_2), ..., Y = k if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = 0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=0) = 0
# h(0) = 0 ---> do not need to model
# S(0) = P(Y > 0 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 1,2,...,k
corresponding Y = 1, ..., k. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
# def neg_likelihood_loss(hazards, Y, c):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
#   # without padding, S(1) = S[0], h(1) = h[0]
#   S_padded = torch.cat([torch.ones_like(c), S], 1) #S(0) = 1, all patients are alive from (-inf, 0) by definition
#   # after padding, S(0) = S[0], S(1) = S[1], etc, h(1) = h[0]
#   #h[y] = h(1)
#   #S[1] = S(1)
#   neg_l = - c * torch.log(torch.gather(S_padded, 1, Y)) - (1 - c) * (torch.log(torch.gather(S_padded, 1, Y-1)) + torch.log(hazards[:, Y-1]))
#   neg_l = neg_l.mean()
#   return neg_l


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = -1 if T_cont \in (-inf, 0), Y = 0 if T_cont \in [0, a_1),  Y = 1 if T_cont in [a_1, a_2), ..., Y = k-1 if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = -1,0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
# h(-1) = 0 ---> do not need to model
# S(-1) = P(Y > -1 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)