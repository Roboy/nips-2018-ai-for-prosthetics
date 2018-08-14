import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable


class NoisyLinear(Module):
    """Applies a noisy linear transformation to the incoming data:
    :math:`y = (mu_w + sigma_w \cdot epsilon_w)x + mu_b + sigma_b \cdot epsilon_b`
    More details can be found in the paper `Noisy Networks for Exploration` _ .
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None,
            defaults to 0.017 for independent and 0.4 for factorised. Default: None
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = nn.NoisyLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True, factorised=True, std_init=None):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if not std_init:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        x = torch.Tensor(size).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.factorised:
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)
            weight_epsilon = Variable(epsilon_out.ger(epsilon_in))
            bias_epsilon = Variable(self.scale_noise(self.out_features))
        else:
            weight_epsilon = Variable(torch.Tensor(self.out_features, self.in_features).normal_())
            bias_epsilon = Variable(torch.Tensor(self.out_features).normal_())
        return F.linear(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
