import torch
import torch.nn.functional as F
import numpy as np

@torch.jit.script
def sample_normal_jit(mu, sigma):
    rho = mu.mul(0).normal_()
    z = rho.mul_(sigma).add_(mu)
    return z, rho


@torch.jit.script
def log_p_standard_normal(samples):
    log_p = - 0.5 * torch.square(samples) - 0.9189385332  # 0.5 * np.log(2 * np.pi)
    return log_p


class Normal:
    def __init__(self, mu, log_var):
        self.mu = mu
        self.log_var = log_var
        self.sigma = torch.exp(0.5*log_var)
        self.log_sigma = torch.log(self.sigma)

    def sample(self, t=1.):
        return sample_normal_jit(self.mu, self.sigma * t)

    def sample_given_rho(self, rho):
        return rho * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - self.log_sigma
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(self.log_sigma) + normal_dist.log_sigma

    def mean(self):
        return self.mu