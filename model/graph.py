import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGATLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.attn_linear = nn.Linear(self.head_dim, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, adjacency_matrix):
        batch_size, num_nodes, hidden_dim = node_features.shape

        # Add self-loops to adjacency matrix
        identity_matrix = torch.eye(num_nodes, device=adjacency_matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adjacency_matrix_with_self_loops = adjacency_matrix + identity_matrix

        # Normalize adjacency matrix
        degree_matrix = adjacency_matrix_with_self_loops.sum(dim=-1, keepdim=True)
        adjacency_matrix_with_self_loops = adjacency_matrix_with_self_loops / degree_matrix

        # Linear transformations
        query = self.query_linear(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        key = self.key_linear(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        value = self.value_linear(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Transpose for multi-head attention
        query = query.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_nodes, head_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # Compute attention scores
        attention_scores = torch.einsum("bhid,bhjd->bhij", query, key) / self.head_dim**0.5  # Scaled dot-product
        attention_scores = attention_scores.masked_fill(adjacency_matrix_with_self_loops.unsqueeze(1) == 0, float('-inf'))
        attention_scores = self.leaky_relu(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Aggregate neighbor features
        aggregated_features = torch.einsum("bhij,bhjd->bhid", attention_weights, value)  # (batch_size, num_heads, num_nodes, head_dim)

        # Concatenate heads and project
        aggregated_features = aggregated_features.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, hidden_dim)
        aggregated_features = self.output_linear(aggregated_features)

        return aggregated_features


class GCN(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, num_heads=4, dropout=0.1):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # Define multi-head GAT layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gat": MultiHeadGATLayer(hidden_dim, num_heads, dropout),
                "feed_forward": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])

    def forward(self, node_features, adjacency_matrix):
        for layer in self.layers:
            # Apply GAT layer
            aggregated_features = layer["gat"](node_features, adjacency_matrix)

            # Residual connection and normalization
            h = layer["norm1"](node_features + aggregated_features)
            h = layer["norm2"](h + layer["feed_forward"](h))

            # Update node features
            node_features = h

        if torch.isnan(node_features).any():
            print("!!!! NAN detected in MultiHeadGAT")
            print("Node features before aggregation:", node_features)
            print("Adjacency matrix:", adjacency_matrix)

        return node_features
