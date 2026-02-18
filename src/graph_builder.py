import torch
import numpy as np
from torch_geometric.data import Data


def build_graph(embeddings, attention, top_k=5, local_k=10):

    L = embeddings.shape[0]
    edge_list = []

    # Local edges
    for i in range(L):
        for j in range(max(0, i - local_k), min(L, i + local_k + 1)):
            if i != j:
                edge_list.append((i, j))

    # Long-range attention edges
    attn_np = attention.numpy()

    for i in range(L):
        topk = np.argsort(attn_np[i])[-top_k:]
        for j in topk:
            if abs(i - j) > 5:
                edge_list.append((i, j))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(
        x=embeddings.float(),
        edge_index=edge_index
    )

    return data
