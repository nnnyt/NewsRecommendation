import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, nb_head, input_dim):
        super(SelfAttention, self).__init__()
        assert input_dim % nb_head == 0
        self.nb_head = nb_head
        self.head_dim = int(input_dim / nb_head)
        self.dim = input_dim
        self.Q = nn.ParameterList([
            nn.Parameter(
                torch.empty(input_dim, input_dim).uniform_(-0.1, 0.1))
            for _ in range(nb_head)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(
                torch.empty(input_dim, self.head_dim).uniform_(-0.1, 0.1))
            for _ in range(nb_head)
        ])
    
    def forward(self, x):
        # batch_size, x_size, x_dim
        out_list = []
        for i in range(self.nb_head):
            # batch_size, x_size, x_dim
            temp = torch.matmul(x, self.Q[i])
            # batch_size, x_size, x_size
            temp = torch.bmm(temp, x.transpose(1, 2))
            # batch_size, x_size, x_size
            att_weights = F.softmax(temp, dim=2)
            # batch_size, x_size, x_dim
            weighted = torch.bmm(att_weights, x)
            # batch_size, x_size, head_dim
            out_list.append(torch.matmul(weighted, self.V[i]))
        # batch_size, x_size, head_dim*nb_head
        output = torch.cat(out_list, dim=2)
        return output
        