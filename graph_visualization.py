import plotly.graph_objects as go
import numpy as np
import torch


def create_3d_residue_graph(
    edge_index,
    residue_importance,
    num_residues_to_show=20,
    sequence_length=None
):
    """
    Create a 3D graph visualization of residues and their connections.
    
    Args:
        edge_index: Tensor of shape [2, num_edges] with edge connections
        residue_importance: Array of importance scores per residue (numpy or tensor)
        num_residues_to_show: Maximum number of residues to visualize
        sequence_length: Total sequence length (for reference)
    
    Returns:
        Plotly figure object
    """
    
    # Convert to numpy if tensor
    if isinstance(residue_importance, torch.Tensor):
        importance_np = residue_importance.cpu().numpy()
    else:
        importance_np = residue_importance
    
    # Get top N important residues
    num_nodes = len(importance_np)
    num_to_show = min(num_residues_to_show, num_nodes)
    
    top_indices = np.argsort(importance_np)[-num_to_show:]
    top_indices_set = set(top_indices.tolist())
    
    # Filter edges to only include residues we're showing
    edge_list = edge_index.cpu().numpy()
    valid_edges = []
    for i in range(edge_list.shape[1]):
        src, dst = edge_list[0, i], edge_list[1, i]
        if src in top_indices_set and dst in top_indices_set:
            valid_edges.append((src, dst))
    
    valid_edges = np.array(valid_edges).T
    
    # Create 3D coordinates for nodes
    node_positions = {}
    for idx, res_idx in enumerate(top_indices):
        res_idx = int(res_idx)
        importance = importance_np[res_idx]
        
        # Assign 3D coordinates
        x = float(res_idx)  # Position in sequence
        y = float(importance * 10)  # Height based on importance
        z = float(np.cos(res_idx / num_nodes * 2 * np.pi) * 5)  # Circular spread
        
        node_positions[res_idx] = (x, y, z)
    
    # Extract edge coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    
    if len(valid_edges) > 0:
        for src, dst in valid_edges.T:
            src, dst = int(src), int(dst)
            x0, y0, z0 = node_positions[src]
            x1, y1, z1 = node_positions[dst]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
    
    # Extract node coordinates
    node_x = []
    node_y = []
    node_z = []
    node_colors = []
    node_text = []
    
    for res_idx in sorted(top_indices):
        x, y, z = node_positions[res_idx]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        importance_val = importance_np[res_idx]
        node_colors.append(importance_val)
        node_text.append(f"Residue {res_idx}<br>Importance: {importance_val:.4f}")
    
    # Create trace for edges
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightblue', width=2),
        hoverinfo='none',
        name='Connections'
    )
    
    # Create trace for nodes
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=8,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Importance",
                thickness=15,
                len=0.7
            ),
            line=dict(color='darkblue', width=2)
        ),
        text=[f"R{int(idx)}" for idx in top_indices],
        textposition="top center",
        textfont=dict(size=8),
        name='Residues'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title={
            'text': f"3D Residue Interaction Network (Top {num_to_show} by Importance)",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        scene=dict(
            xaxis=dict(title='Sequence Position', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Importance Score', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='Spatial Distribution', backgroundcolor="rgb(230, 230,230)"),
            bgcolor="rgb(240, 240, 240)"
        ),
        width=1200,
        height=700
    )
    
    return fig
