"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform,SigmoidTransform,AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

# """Additive coupling layer.
# """
#
#
# class AdditiveCoupling(nn.Module):
#     def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
#         """Initialize an additive coupling layer.
#
#         Args:
#             in_out_dim: input/output dimensions.
#             mid_dim: number of units in a hidden layer.
#             hidden: number of hidden layers.
#             mask_config: 1 if transform odd units, 0 if transform even units.
#         """
#         super(AdditiveCoupling, self).__init__()
#         self.mask_config = mask_config
#         self.in_block = nn.Sequential(nn.ReLU(nn.Linear(in_out_dim//2, mid_dim)))
#         self.mid_block = nn.ModuleList([nn.Sequential(nn.ReLU(nn.Linear(mid_dim, mid_dim)))\
#                                         for _ in range(hidden-1)])
#         self.out_block = nn.Linear(mid_dim, in_out_dim//2)
#
#     def forward(self, x, log_det_J, reverse=False):
#         """Forward pass.
#
#         Args:
#             x: input tensor.
#             log_det_J: log determinant of the Jacobian
#             reverse: True in inference mode, False in sampling mode.
#         Returns:
#             transformed tensor and log-determinant of Jacobian.
#         """
#         [B, W] = list(x.size())
#         x = x.view((B,W//2,2))
#         if self.mask_config:
#             on, off = x[:, :, 0], x[:, :, 1]
#         else:
#             on, off = x[:, :, 1], x[:, :, 0]
#
#         off_ = self.in_block(off)
#         for i in range(len(self.mid_block)):
#             off_ = self.mid_block[i](off_)
#         shift = self.out_block(off_)
#
#         if reverse:
#             on = on - shift
#         else:
#             on = on + shift
#
#         if self.mask_config:
#             x = torch.stack((on, off), dim=2)
#         else:
#             x = torch.stack((off, on), dim=2)
#
#         return x.view((B, W)), log_det_J
#
#
# class AffineCoupling(nn.Module):
#     def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
#         """Initialize an affine coupling layer.
#
#         Args:
#             in_out_dim: input/output dimensions.
#             mid_dim: number of units in a hidden layer.
#             hidden: number of hidden layers.
#             mask_config: 1 if transform odd units, 0 if transform even units.
#         """
#         super(AffineCoupling, self).__init__()
#         self.mask_config = mask_config
#         self.in_block = nn.Sequential(nn.ReLU(nn.Linear(in_out_dim//2, mid_dim)))
#         self.mid_block = nn.ModuleList([nn.Sequential(nn.ReLU(nn.Linear(mid_dim, mid_dim))) \
#                                         for _ in range(hidden-1)])
#         self.out_block = nn.Linear(mid_dim, in_out_dim//2)
#
#     def forward(self, x, log_det_J, reverse=False):
#         """Forward pass.
#
#         Args:
#             x: input tensor.
#             log_det_J: log determinant of the Jacobian
#             reverse: True in inference mode, False in sampling mode.
#         Returns:
#             transformed tensor and log-determinant of Jacobian.
#         """
#
#         [B, W] = list(x.size())
#         x = x.view((B,W//2,2))
#         if self.mask_config:
#             on, off = x[:, :, 0], x[:, :, 1]
#         else:
#             on, off = x[:, :, 1], x[:, :, 0]
#
#         off_ = self.in_block(off)
#         for i in range(len(self.mid_block)):
#             off_ = self.mid_block[i](off_)
#         shift = self.out_block(off_)
#         shift_1, shift_2 = shift.chunk(2,1)
#         if reverse:
#             on = on / shift_1 - shift_2
#         else:
#             on = shift_1 * on + shift_2
#
#         if self.mask_config:
#             x = torch.stack((on, off), dim=2)
#         else:
#             x = torch.stack((off, on), dim=2)
#         log_det_J = torch.sum(torch.log(shift_1).view(input.shape[0], -1), 1)
#         return x.view((B, W)), log_det_J
#
# """Log-scaling layer.
# """
#
#
# class Scaling(nn.Module):
#     def __init__(self, dim):
#         """Initialize a (log-)scaling layer.
#
#         Args:
#             dim: input/output dimensions.
#         """
#         super(Scaling, self).__init__()
#         self.scale = nn.Parameter(
#             torch.zeros((1, dim)), requires_grad=True)
#         self.eps = 1e-5
#
#     def forward(self, x, reverse=False):
#         """Forward pass.
#
#         Args:
#             x: input tensor.
#             reverse: True in inference mode, False in sampling mode.
#         Returns:
#             transformed tensor and log-determinant of Jacobian.
#         """
#         log_det_J = torch.sum(self.scale)
#         if reverse:
#             scale = torch.exp(-self.scale)+ self.eps
#         else:
#             scale = torch.exp(self.scale) + self.eps
#         x *= scale
#         return x, log_det_J
#
#
# """NICE main model.
# """
#
#
# class NICE(nn.Module):
#     def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config,device, coup_type = "additive"):
#         """Initialize a NICE.
#
#         Args:
#             coupling: number of coupling layers.
#             in_out_dim: input/output dimensions.
#             mid_dim: number of units in a hidden layer.
#             hidden: number of hidden layers.
#             mask_config: 1 if transform odd units, 0 if transform even units.
#         """
#         super(NICE, self).__init__()
#         self.device = device
#         if prior == 'gaussian':
#             self.prior = torch.distributions.Normal(
#                 torch.tensor(0.).to(device), torch.tensor(1.).to(device))
#         elif prior == 'logistic':
#             self.prior = TransformedDistribution(Uniform(0, 1),
#                                                  [SigmoidTransform().inv,
#                                                   AffineTransform(loc=0., scale=1.)])
#         else:
#             raise ValueError('Prior not implemented.')
#         self.in_out_dim = in_out_dim
#         if coup_type == "additive":
#             self.coupling = nn.ModuleList([AdditiveCoupling(in_out_dim=in_out_dim, mid_dim=mid_dim,
#                                                             hidden=hidden, mask_config=(mask_config+i)%2)\
#                                            for i in range(coupling)])
#         else:
#             self.coupling = nn.ModuleList([AffineCoupling(in_out_dim=in_out_dim, mid_dim=mid_dim,
#                                                           hidden=hidden, mask_config=(mask_config+i)%2)\
#                                            for i in range(coupling)])
#         self.scaling = Scaling(in_out_dim)
#
#     def f_inverse(self, z):
#         """Transformation g: Z -> X (inverse of f).
#
#         Args:
#             z: tensor in latent space Z.
#         Returns:
#             transformed tensor in data space X.
#         """
#         x, log_det_J = self.scaling(z, reverse=True)
#         for i in reversed(range(len(self.coupling))):
#             x, log_det_J = self.coupling[i](x, log_det_J, reverse = True)
#         return x, log_det_J
#
#     def f(self, x):
#         """Transformation f: X -> Z (inverse of g).
#
#         Args:
#             x: tensor in data space X.
#         Returns:
#             transformed tensor in latent space Z and log determinant Jacobian
#         """
#         log_det_J = 0
#         for i in range(len(self.coupling)):
#             x, log_det_J = self.coupling[i](x, log_det_J)
#         return self.scaling(x)
#
#     def log_prob(self, x):
#         """Computes data log-likelihood.
#
#         (See Section 3.3 in the NICE paper.)
#
#         Args:
#             x: input minibatch.
#         Returns:
#             log-likelihood of input.
#         """
#         z, log_det_J = self.f(x)
#         log_det_J -= np.log(256)*self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
#         log_ll = torch.sum(self.prior.log_prob(z), dim=1)
#         return log_ll + log_det_J
#
#     def sample(self, size):
#         """Generates samples.
#
#         Args:
#             size: number of samples to generate.
#         Returns:
#             samples from the data space X.
#         """
#         z = self.prior.sample((size, self.in_out_dim)).to(self.device)
#         return self.f_inverse(z)
#
#     def forward(self, x):
#         """Forward pass.
#
#         Args:
#             x: input minibatch.
#         Returns:
#             log-likelihood of input.
#         """
#         return self.log_prob(x)
#
#
# import torch
# import torch.nn as nn
#
# """Additive coupling layer.
# """


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling,
                 in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a NICE.
        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList([
            AdditiveCoupling(in_out_dim=in_out_dim,
                     mid_dim=mid_dim,
                     hidden=hidden,
                     mask_config=(mask_config+i)%2) \
            for i in range(coupling)])
        self.scaling = Scaling(in_out_dim)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).
        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)

    def log_prob(self, x):
        """Computes data log-likelihood.
        (See Section 3.3 in the NICE paper.)
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).cuda()
        return self.g(z)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)