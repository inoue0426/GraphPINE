from typing import List, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from matplotlib import cm
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph
from adjustText import adjust_text

from .utils import fetch_drug_name


def visualize_top_nodes_wrapper(
    node_features: pd.DataFrame,
    top_k: int = 100,
    title: str = "Top Important Nodes in Graph",
    layout: str = "spring",
    colorscale: str = "YlOrRd",
) -> List[Union[go.Figure, plt.Figure]]:
    """
    A wrapper function to prepare data and call visualize_top_nodes.
    """
    dti = pd.read_csv("data/dti.csv.gz", index_col=0)
    nsc = list(node_features["nsc"])
    cell_name = list(node_features["cell_name"])
    types = list(node_features["type"])
    node_features = node_features.iloc[:, 4:]

    edge_index = torch.load("data/edge_index.pt", weights_only=False)
    data_list = [
        Data(
            x=torch.tensor(
                node_features.iloc[i].values.astype(float), dtype=torch.float
            ),
            edge_index=edge_index,
            dti=torch.tensor(dti.loc[nsc[i]].values.astype(float), dtype=torch.float),
            nsc=nsc[i],
            cell_name=cell_name[i],
            type=types[i],
        )
        for i in range(node_features.shape[0])
    ]

    data_batch = Batch.from_data_list(data_list)
    importance = torch.tensor(
        node_features.values.astype(float).flatten(), dtype=torch.float
    )

    return visualize_top_nodes(
        data_batch=data_batch,
        importance=importance,
        top_k=top_k,
        title=title,
        layout=layout,
        colorscale=colorscale,
    )


def prepare_node_trace(
    node_x: List[float],
    node_y: List[float],
    node_sizes: List[float],
    node_symbols: List[str],
    importance_np: np.ndarray,
    top_genes: List[str],
    top_dti: torch.Tensor,
    min_importance: float,
    max_importance: float,
    colorscale: str,
) -> Tuple[go.Scatter, go.Scatter]:
    """
    Prepare node traces for DTI and non-DTI nodes.
    """

    def create_node_trace(symbol: str) -> go.Scatter:
        mask = [s == symbol for s in node_symbols]
        return go.Scatter(
            x=[x for x, m in zip(node_x, mask) if m],
            y=[y for y, m in zip(node_y, mask) if m],
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=[size for size, m in zip(node_sizes, mask) if m],
                color=[imp for imp, m in zip(importance_np, mask) if m],
                colorscale=colorscale,
                cmin=min_importance,
                cmax=max_importance,
                line_width=2,
                colorbar=(
                    dict(
                        thickness=15,
                        title="Node Importance",
                        xanchor="left",
                        titleside="right",
                        len=0.5,
                        tickvals=[min_importance, max_importance],
                        ticktext=["Low", "High"],
                    )
                    if symbol == "star"
                    else None
                ),
            ),
            text=[
                f"Gene: {gene}<br>Importance: {imp:.4f}<br>DTI: {dti:.4f}"
                for gene, imp, dti, m in zip(top_genes, importance_np, top_dti, mask)
                if m
            ],
            hoverinfo="text",
            name=f"Nodes {'with' if symbol == 'star' else 'without'} DTI ({len([m for m in mask if m])})",
        )

    return create_node_trace("star"), create_node_trace("circle")


def process_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    importance: torch.Tensor,
    genes: List[str],
    nsc: str,
    cell_name: str,
    data_type: str,
    dti: torch.Tensor,
    layout: str,
    colorscale: str,
    top_k: int,
    title: str,
) -> go.Figure:
    """
    Process a single graph and create a Plotly figure.
    """
    graph_top_k = min(top_k, x.size(0))
    _, top_indices = torch.topk(importance, k=graph_top_k)
    top_x = x[top_indices]
    top_importance = importance[top_indices]
    top_genes = [genes[i] for i in top_indices.tolist()]
    top_dti = dti[top_indices]

    top_mask = torch.zeros_like(importance, dtype=torch.bool)
    top_mask[top_indices] = True
    top_edge_index, top_edge_attr = subgraph(
        top_mask, edge_index, edge_attr, relabel_nodes=True
    )

    G = nx.Graph()
    G.add_edges_from(top_edge_index.t().tolist())

    layout_func = getattr(nx, f"{layout}_layout", nx.spring_layout)
    pos = layout_func(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    node_x, node_y = zip(*[pos[node] for node in G.nodes()])
    importance_np = top_importance.detach().cpu().numpy()
    min_size, max_size = 10, 50
    node_sizes = min_size + (importance_np - importance_np.min()) / (
        importance_np.max() - importance_np.min()
    ) * (max_size - min_size)
    node_symbols = ["star" if dti != 0 else "circle" for dti in top_dti]

    dti_nodes, non_dti_nodes = prepare_node_trace(
        node_x,
        node_y,
        node_sizes,
        node_symbols,
        importance_np,
        top_genes,
        top_dti,
        importance_np.min(),
        importance_np.max(),
        colorscale,
    )

    drug_name = fetch_drug_name(nsc)

    return go.Figure(
        data=[edge_trace, dti_nodes, non_dti_nodes],
        layout=go.Layout(
            title=dict(
                text=f"{title}<br>(Drug: {drug_name}, Cell: {cell_name}, Type: {data_type})",
                y=0.95,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            titlefont_size=16,
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[
                dict(
                    text=f"Top {graph_top_k} nodes out of {x.size(0)}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="black"),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2,
            ),
        ),
    )


def visualize_top_nodes(
    data_batch: Union[Data, Batch],
    importance: torch.Tensor,
    top_k: int = 100,
    title: str = "Top Important Nodes in Graph",
    layout: str = "spring",
    colorscale: str = "YlOrRd",
) -> List[go.Figure]:
    """
    Visualize the top K important nodes from each graph in a DataBatch object.
    """
    genes = list(pd.read_csv("data/genes.csv").columns)

    if importance.dim() > 1:
        importance = importance.squeeze()
    assert data_batch.num_nodes == importance.size(
        0
    ), "Number of nodes in data_batch and importance must match"

    if hasattr(data_batch, "batch"):
        num_graphs = data_batch.batch.max().item() + 1
        return [
            process_graph(
                data_batch.x[data_batch.batch == i],
                *subgraph(
                    data_batch.batch == i,
                    data_batch.edge_index,
                    data_batch.edge_attr,
                    relabel_nodes=True,
                ),
                importance[data_batch.batch == i],
                genes,
                data_batch.nsc[i],
                data_batch.cell_name[i],
                data_batch.type[i],
                data_batch.dti[data_batch.batch == i],
                layout,
                colorscale,
                top_k,
                title,
            )
            for i in range(num_graphs)
        ]
    else:
        return [
            process_graph(
                data_batch.x,
                data_batch.edge_index,
                data_batch.edge_attr,
                importance,
                genes,
                data_batch.nsc,
                data_batch.cell_name,
                data_batch.type,
                data_batch.dti,
                layout,
                colorscale,
                top_k,
                title,
            )
        ]


def plot_training_curves(train_metrics: List[dict], val_metrics: List[dict]):
    """
    Plot training and validation curves for Loss, Accuracy, and Precision.
    """
    num_epochs = len(train_metrics)
    epochs = range(1, num_epochs + 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Training and Validation Metrics over Epochs", fontsize=16)

    metrics = ["loss", "accuracy", "precision"]
    titles = ["Loss", "Accuracy", "Precision"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axs[i]
        train_values = [m[metric] for m in train_metrics]
        val_values = [m[metric] for m in val_metrics]

        ax.plot(epochs, train_values, label=f"Training {title}")
        ax.plot(epochs, val_values, label=f"Validation {title}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(title)
        ax.set_title(f"{title} over Epochs")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_gene_interaction_network(
    drug: str,
    N: int = 20,
    res: pd.DataFrame = None,
    font_size: int = 10,
    min_size: int = 300,
    max_size: int = 3000,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
    seed: int = 42,
    shrink: float = 0.4,
    pad: float = 0.02,
    orientation: str = "vertical",
    loc: str = "lower center",
    bbox_to_anchor: Tuple[float, float] = (0.5, -0.22),
    layout: str = "spring",
):
    """
    Plot the gene interaction network for a given drug with top N genes highlighted.
    """
    # Setting consistent font settings
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size

    # Load data
    edge_index = torch.load("data/edge_index.pt", weights_only=False)
    dti = pd.read_csv("data/dti.csv.gz", index_col=0)

    # Filter data for the given drug
    k = res[res["name"] == drug].iloc[:, 4:]
    dti = dti.loc[res[res["name"] == drug]["nsc"].iloc[0]]

    # Identify the top N genes with the highest values
    top_genes = k.T.nlargest(N, k.index[0]).index.tolist()

    # Map gene names to node indices and vice versa
    gene_to_node_mapping = {gene: idx for idx, gene in enumerate(k.columns)}
    node_to_gene_mapping = {idx: gene for gene, idx in gene_to_node_mapping.items()}

    # Get the corresponding node indices for top genes
    top_node_indices = [gene_to_node_mapping[gene] for gene in top_genes]

    # Filter edges to include only those where both nodes are in the top nodes
    filtered_edges = [
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(edge_index.size(1))
        if edge_index[0, i].item() in top_node_indices
        and edge_index[1, i].item() in top_node_indices
    ]

    # Create the graph
    G = nx.Graph()
    G.add_edges_from(filtered_edges)

    # Add isolated nodes if they are in top_genes but not in the graph
    for gene in top_genes:
        if gene not in G.nodes():
            G.add_node(gene)

    # Rename nodes in the graph with the gene names
    G = nx.relabel_nodes(G, node_to_gene_mapping)

    # Calculate node sizes based on the corresponding values in DataFrame `k`
    raw_node_sizes = [k[node].iloc[0] * 1000 for node in G.nodes]

    # Normalize the node sizes
    normalized_node_sizes = np.interp(
        raw_node_sizes, (min(raw_node_sizes), max(raw_node_sizes)), (min_size, max_size)
    )

    # Normalize dti values to a range [0, 1] with the highest value being 1
    dti = dti[dti.index.isin(top_genes)]
    max_dti = dti.max()
    normalized_dti = dti / max_dti if max_dti != 0 else dti
    vmin, vmax = 0, 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd  # Use the YlOrRd colormap
    node_colors = [cmap(normalized_dti[node]) for node in G.nodes]

    # Count known and unknown drug-target interactions
    known_dti = sum(1 for node in G.nodes if dti[node] > 0)
    unknown_dti = len(G.nodes) - known_dti

    # Draw the graph with normalized node sizes and colors based on dti values
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Choose layout based on the layout parameter
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G, seed=seed)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)  # Default to spring layout
    
    nx.draw_networkx_edges(
        G, pos, alpha=0.5, edge_color="gray", width=2, ax=ax
    )  # Set edge properties
    
    # Draw nodes with colors and shapes based on DTI
    for node, (x, y) in pos.items():
        if dti[node] > 0:
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node], node_color=[node_colors[list(G.nodes).index(node)]],
                node_size=normalized_node_sizes[list(G.nodes).index(node)],
                ax=ax, node_shape='*', edgecolors='black', linewidths=1
            )
        else:
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node], node_color=[node_colors[list(G.nodes).index(node)]],
                node_size=normalized_node_sizes[list(G.nodes).index(node)],
                ax=ax, node_shape='o', edgecolors='black', linewidths=1
            )

    label_pos = {k: (v[0], v[1] + 0.13) for k, v in pos.items()}
    label_pos['TOP1'] = (pos['TOP1'][0], pos['TOP1'][1] + 0.3)

    # ラベルの描画（背景付き）
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', ax=ax,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
    
    ax.set_title(f"{drug}-Related Gene Interaction Network", fontsize=font_size)

    # Add colorbar using the axes of the plot
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=shrink, pad=pad, orientation=orientation)
    cbar.set_label("DTI score", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # Modify colorbar ticks and labels
    cbar.set_ticks([0, vmax])
    cbar.set_ticklabels(["Low", "High"])

    # Add legend
    markersize = 10

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Propagated Importance from GNN training (Size)",
            markerfacecolor="gray",
            markersize=markersize,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Drug-Target Interaction score (Color)",
            markerfacecolor=cmap(0.5),
            markersize=markersize,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Unknown Drug-Target Interaction (N: {unknown_dti})",
            markerfacecolor="gray",
            markersize=markersize,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label=f"Known Drug-Target Interaction (N: {known_dti})",
            markerfacecolor="gray",
            markersize=markersize * 1.5,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc=loc,
        fontsize=font_size,
        bbox_to_anchor=bbox_to_anchor,
    )

    plt.tight_layout()
    plt.savefig(f"result/interpret_{drug}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


# Example usage
# plot_gene_interaction_network('Vemurafenib', N=20)
