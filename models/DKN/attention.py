import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """ Additive Attention """
    def __init__(self, attention_dim, input_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.q = nn.Parameter(torch.zeros(attention_dim).uniform_(-0.1, 0.1))
        self.linear = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        # x: batch_size, x_size, x_dim
        temp = torch.tanh(self.linear(x))
        # batch_size, x_size
        att_weights = F.softmax(torch.matmul(temp, self.q),dim=1)
        # batch_size, x_dim
        output = torch.bmm(att_weights.unsqueeze(dim=1),
                           x).squeeze(dim=1)
        return output

