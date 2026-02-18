import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.model import ProteinGAT


def load_gat_model(model_path, device="cpu"):
    model = ProteinGAT()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model


# def predict(model, graph_data, idx_to_go, device="cpu", top_k=5):
#     graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)

#     graph_data = graph_data.to(device)

#     with torch.no_grad():
#         logits = model(graph_data)
#         probs = torch.sigmoid(logits)

#     probs = probs.squeeze().cpu().numpy()

#     top_indices = probs.argsort()[-top_k:][::-1]

#     results = []
#     for idx in top_indices:
#         results.append((idx_to_go[idx], float(probs[idx])))

#     return results

import torch
import torch.nn.functional as F


def predict(model, graph_data, idx_to_go, device="cpu", top_k=5):
    model.eval()

    # Create batch vector (single protein)
    graph_data.batch = torch.zeros(
        graph_data.x.shape[0],
        dtype=torch.long
    )

    graph_data = graph_data.to(device)

    with torch.no_grad():
        logits, edge_index, attn_weights = model(
            graph_data,
            return_attention=True
        )

    probs = torch.sigmoid(logits)[0]

    topk = torch.topk(probs, top_k)

    results = []
    for idx, score in zip(topk.indices, topk.values):
        go_term = idx_to_go[idx.item()]
        results.append((go_term, float(score)))

    return results, edge_index.cpu(), attn_weights.cpu()

def compute_residue_importance(edge_index, attn_weights, num_nodes):
    importance = torch.zeros(num_nodes)

    # If attention shape is [num_edges, 1]
    if attn_weights.dim() > 1:
        attn_weights = attn_weights.squeeze()

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        weight = attn_weights[i]

        importance[src] += weight
        importance[dst] += weight

    importance = importance / importance.max()

    return importance.numpy()
