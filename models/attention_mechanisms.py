from torch import nn
from torch.nn.functional import softmax
import torch


class SelfAttention(nn.Module):
    """
    A self-attention layer for use in a multi-head self-attention LSTM network.

    Args:
        input_dim (int): The dimensionality of the input.
        output_dim (int): The dimensionality of the output.

    Attributes:
        W_query (nn.Linear): The linear layer for the query projection.
        W_key (nn.Linear): The linear layer for the key projection.
        W_value (nn.Linear): The linear layer for the value projection.
    """
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.W_query = nn.Linear(input_dim, output_dim)
        self.W_key = nn.Linear(input_dim, output_dim)
        self.W_value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attention_weights = softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MultiHeadSelfAttention(nn.Module):
    """
    A multi-head self-attention layer for use in a multi-head self-attention LSTM network.

    Args:
        input_dim (int): The dimensionality of the input.
        output_dim (int): The dimensionality of the output.
        num_heads (int): The number of attention heads to use.

    Attributes:
        attention_heads (nn.ModuleList): A list of SelfAttention layers.
        linear (nn.Linear): The linear layer used to combine the attention outputs.
    """
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_heads = nn.ModuleList([SelfAttention(input_dim, output_dim) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * output_dim, input_dim)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        concatenated_outputs = torch.cat(attention_outputs, dim=-1)
        return self.linear(concatenated_outputs)
