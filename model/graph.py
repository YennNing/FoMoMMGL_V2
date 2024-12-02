import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(GATLayer, self).__init__()
        self.attn_linear = nn.Linear(2 * hidden_dim, 1)  
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adjacency_matrix):
        batch_size, num_nodes, hidden_dim = node_features.shape

        # Add self-loops to adjacency matrix
        identity_matrix = torch.eye(num_nodes, device=adjacency_matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adjacency_matrix_with_self_loops = adjacency_matrix + identity_matrix

        # Broadcast features for pairwise concatenation (node_i || neighbor_j)
        node_features_expanded = node_features.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # (batch_size, num_nodes, num_nodes, hidden_dim)
        neighbor_features_expanded = node_features.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # (batch_size, num_nodes, num_nodes, hidden_dim)

        # Concatenate features for attention computation
        concat_features = torch.cat([node_features_expanded, neighbor_features_expanded], dim=-1)  # (batch_size, num_nodes, num_nodes, 2 * hidden_dim)

        # Compute attention scores
        attention_scores = self.attn_linear(concat_features).squeeze(-1)  # (batch_size, num_nodes, num_nodes)
        attention_scores = self.leaky_relu(attention_scores)

        # Mask with adjacency matrix (with self-loops) and normalize (softmax over neighbors)
        attention_scores = attention_scores.masked_fill(adjacency_matrix_with_self_loops == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_nodes, num_nodes)
        attention_weights = self.dropout(attention_weights)

        # Aggregate neighbor features using attention weights
        aggregated_features = torch.bmm(attention_weights, node_features)  # (batch_size, num_nodes, hidden_dim)

        return aggregated_features


class GCN(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, dropout=0.1):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # Define GAT layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gat": GATLayer(hidden_dim, dropout),
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
            print("!!!! NAN detected in GCN")
            print("Node features before aggregation:", node_features)
            print("Adjacency matrix:", adjacency_matrix)

        return node_features
