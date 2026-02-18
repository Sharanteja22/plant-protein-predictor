# import torch
# import torch.nn as nn
# from torch_geometric.nn import GATConv, global_mean_pool


# class ProteinGAT(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=256, out_dim=323):
#         super().__init__()

#         self.gat = GATConv(
#             in_channels=in_dim,
#             out_channels=hidden_dim,
#             heads=1,
#             concat=False
#         )

#         self.norm = nn.LayerNorm(hidden_dim)

#         self.classifier = nn.Linear(hidden_dim, out_dim)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         h = self.gat(x, edge_index)
#         h = self.norm(h + x)  # residual

#         h_protein = global_mean_pool(h, batch)
#         out = self.classifier(h_protein)

#         return out
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class ProteinGAT(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=323):
        super().__init__()

        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data, return_attention=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if return_attention:
            h, (edge_index_out, attn_weights) = self.gat(
                x,
                edge_index,
                return_attention_weights=True
            )
        else:
            h = self.gat(x, edge_index)

        # Residual connection
        h = self.norm(h + x)

        # Pool residues → protein vector
        h_protein = global_mean_pool(h, batch)

        out = self.classifier(h_protein)

        if return_attention:
            return out, edge_index_out, attn_weights
        else:
            return out
