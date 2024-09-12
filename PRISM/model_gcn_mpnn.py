import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, MessagePassing

from .utils import free_memory


class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network Layer for the GeneGNN, without using edge attributes.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin_message = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.lin_message(x_j)

    def update(self, aggr_out, x):
        return self.lin_update(torch.cat([x, aggr_out], dim=-1))


class ImportancePropagationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, layer_type):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "mpnn":
            self.conv = MPNNLayer(in_channels, out_channels)
        elif layer_type == "gcn":
            self.conv = GCNConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.importance_gate = nn.Linear(out_channels + 1, out_channels)
        self.importance_propagation = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, importance):
        conv_out = self.conv(x, edge_index)

        gate_input = torch.cat([conv_out, importance], dim=-1)
        gate = torch.sigmoid(self.importance_gate(gate_input))
        out = gate * conv_out + (1 - gate) * x
        propagated_importance = self.importance_propagation(out)
        free_memory()
        return out, propagated_importance


class GeneGNN(nn.Module):
    def __init__(
        self,
        config,
    ):
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
                    config["layer_type"],
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
        importance = self.normalize_importance(initial_importance.squeeze(), batch)

        importance_emb = self.importance_projection(importance.unsqueeze(-1))
        x = self.initial_proj(torch.cat([x, importance_emb], dim=-1))

        for prop_layer, norm in zip(self.prop_layers, self.norms):
            x, propagated_importance = prop_layer(
                x, edge_index, importance.unsqueeze(-1)
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
