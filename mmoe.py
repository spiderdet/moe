# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from mlp import MLP,MLP2
from moe import SparseDispatcher, MoE

class MmoE(nn.Module):
    def __init__(self, input_size, moe_output_size, output_size, num_experts, moe_hidden_size,mlp_hidden_size, noisy_gating=True, k=4):
        super(MmoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.moe_output_size = moe_output_size
        self.moe_hidden_size = moe_hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.k = k
        # instantiate experts
        self.moe = MoE(self.input_size,self.moe_output_size,self.num_experts,self.moe_hidden_size,self.noisy_gating,self.k)
        self.mlp1 = MLP2(self.moe_output_size, self.mlp_hidden_size, self.output_size)
        self.mlp2 = MLP2(self.moe_output_size, self.mlp_hidden_size, self.output_size)
        self.softmax = nn.Softmax(1)

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        moe_x,moe_loss = self.moe(x,loss_coef)
        y1 = self.mlp1(moe_x)
        y2 = self.mlp2(moe_x)
        return y1, y2
