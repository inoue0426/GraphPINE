import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm

from .utils import free_memory


class ImportancePropagationLayer(nn.Module):
    """
    A custom layer for propagating importance through a graph neural network.

    This layer combines GINEConv with an importance gating mechanism
    to update node features and propagate importance values.

    Attributes:
        conv (nn.Module): Convolution layer for processing node features (GINEConv).
        importance_gate (nn.Linear): Linear layer for importance gating.
        importance_propagation (nn.Linear): Linear layer for propagating importance.
    """

    def __init__(self, in_channels, out_channels, edge_dim):
        """
        Initialize the ImportancePropagationLayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            edge_dim (int): Dimension of edge features.
        """
        super().__init__()

        self.conv = GINEConv(nn.Linear(in_channels, out_channels), edge_dim=edge_dim)

        self.importance_gate = nn.Linear(out_channels + 1, out_channels)
        self.importance_propagation = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr, importance):
        """
        Forward pass of the ImportancePropagationLayer.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            edge_attr (torch.Tensor): Edge feature matrix.
            importance (torch.Tensor): Node importance values.

        Returns:
            tuple: Updated node features and propagated importance values.
        """
        conv_out = self.conv(x, edge_index, edge_attr)
        gate_input = torch.cat([conv_out, importance], dim=-1)
        gate = torch.sigmoid(self.importance_gate(gate_input))
        out = gate * conv_out + (1 - gate) * x
        propagated_importance = self.importance_propagation(out)
        free_memory()
        return out, propagated_importance


class GeneGNN(nn.Module):
    """
    Gene Graph Neural Network for processing gene interaction networks.

    This module implements a graph neural network that processes gene interaction
    data, incorporating importance values for each gene. It uses multiple
    ImportancePropagationLayers to update gene features and importance values.

    Attributes:
        num_layers (int): Number of ImportancePropagationLayers.
        importance_decay (float): Decay factor for importance values.
        importance_threshold (float): Threshold for pruning low importance values.
        importance_projection (nn.Linear): Projects importance values to hidden dimension.
        initial_proj (nn.Linear): Initial projection of input features.
        prop_layers (nn.ModuleList): List of ImportancePropagationLayers.
        norms (nn.ModuleList): List of GraphNorm layers.
        dropout (nn.Dropout): Dropout layer.
        final_proj (nn.Linear): Final projection layer.
    """

    def __init__(self, config):
        """
        Initialize the GeneGNN.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            edge_dim (int): Dimension of edge features.
            num_layers (int): Number of ImportancePropagationLayers.
            dropout_rate (float): Dropout rate.
            importance_decay (float): Decay factor for importance values.
            importance_threshold (float): Threshold for pruning low importance values.
        """
        super().__init__()
        self.num_layers = config["NUM_GNN_LAYERS"]
        self.importance_decay = config["IMPORTANCE_DECAY"]
        self.importance_threshold = config["IMPORTANCE_THRESHOLD"]

        self.importance_projection = nn.Linear(1, config["HIDDEN_CHANNEL_SIZE"])
        self.initial_proj = nn.Linear(
            config["GENE_FEATURE_DIMENSION"] + config["HIDDEN_CHANNEL_SIZE"],
            config["HIDDEN_CHANNEL_SIZE"],
        )

        self.prop_layers = nn.ModuleList(
            [
                ImportancePropagationLayer(
                    config["HIDDEN_CHANNEL_SIZE"],
                    config["HIDDEN_CHANNEL_SIZE"],
                    config["GENE_EDGE_DIMENSION"],
                )
                for _ in range(config["NUM_GNN_LAYERS"])
            ]
        )

        self.norms = nn.ModuleList(
            [
                GraphNorm(config["HIDDEN_CHANNEL_SIZE"])
                for _ in range(config["NUM_GNN_LAYERS"])
            ]
        )

        self.dropout = nn.Dropout(p=config["DROPOUT_RATE"])
        self.final_proj = nn.Linear(
            config["HIDDEN_CHANNEL_SIZE"], config["OUTPUT_CHANNEL_SIZE"]
        )

    def normalize_importance(self, importance, batch):
        """
        Normalize importance values within each batch.

        Args:
            importance (torch.Tensor): Importance values.
            batch (torch.Tensor): Batch assignment for each node.

        Returns:
            torch.Tensor: Normalized importance values.
        """
        batch_size = batch.max().item() + 1
        min_val, max_val = torch.zeros(
            batch_size, device=importance.device
        ), torch.zeros(batch_size, device=importance.device)

        for i in range(batch_size):
            mask = batch == i
            min_val[i], max_val[i] = importance[mask].min(), importance[mask].max()

        min_val, max_val = min_val[batch], max_val[batch]
        normalized = (importance - min_val) / (max_val - min_val + 1e-8)
        free_memory()
        return normalized

    def update_importance(self, importance, propagated_importance, batch):
        """
        Update importance values based on propagated importance.

        Args:
            importance (torch.Tensor): Current importance values.
            propagated_importance (torch.Tensor): Propagated importance values.
            batch (torch.Tensor): Batch assignment for each node.

        Returns:
            torch.Tensor: Updated importance values.
        """
        importance = (
            self.importance_decay * importance
            + (1 - self.importance_decay) * propagated_importance.squeeze()
        )
        importance = self.normalize_importance(importance, batch)
        importance = torch.where(
            importance < self.importance_threshold,
            torch.zeros_like(importance),
            importance,
        )
        free_memory()
        return self.normalize_importance(importance, batch)

    def forward(self, x, edge_index, edge_attr, initial_importance, batch):
        """
        Forward pass of the GeneGNN.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            edge_attr (torch.Tensor): Edge feature matrix.
            initial_importance (torch.Tensor): Initial importance values for each node.
            batch (torch.Tensor): Batch assignment for each node.

        Returns:
            tuple: Graph embeddings and final importance values.
        """
        device = x.device
        importance = self.normalize_importance(initial_importance.squeeze(), batch)

        importance_emb = self.importance_projection(importance.unsqueeze(-1))
        x = self.initial_proj(torch.cat([x, importance_emb], dim=-1))

        for prop_layer, norm in zip(self.prop_layers, self.norms):
            x, propagated_importance = prop_layer(
                x, edge_index, edge_attr, importance.unsqueeze(-1)
            )
            x = self.dropout(F.relu(norm(x, batch)))
            importance = self.update_importance(
                importance, propagated_importance, batch
            )
            free_memory()

        x = self.final_proj(x)

        batch_size = batch.max().item() + 1
        graph_embeddings = torch.stack(
            [x[batch == i].mean(dim=0) for i in range(batch_size)]
        )
        logits = graph_embeddings

        final_importance = self.normalize_importance(
            initial_importance.squeeze() + importance, batch
        )

        free_memory()
        return logits, final_importance.unsqueeze(-1)
