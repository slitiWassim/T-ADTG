
import torch
import torch.nn as nn
from torch_geometric.utils import degree

from .layers import (
    History,
    SATGAT,
    GNN,
)
from .time_encoder import TimeEncoder


class TADTG(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, edge_dim: int, config):
        super().__init__()

        self.hidden_dim = config.MODEL.HIDDEN_CHANNELS
        self.heads = config.MODEL.HEADS
        self.num_nodes = num_nodes
        self.dropout = config.MODEL.DROPOUT
        self.momentum=config.TRAIN.MOMENTUM

        self.input_projection = nn.Linear(in_dim,self.hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, self.hidden_dim)
        self.time_encoder = TimeEncoder(self.hidden_dim)
        self.gnn = GNN(dim=self.hidden_dim, heads=self.heads, dropout=self.dropout)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.stru_hist = History(
            num_nodes=num_nodes,
            dim=self.hidden_dim)

        self.layers = nn.ModuleList(
            [
                SATGAT(
                    dim=self.hidden_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                )
                for _ in range(config.MODEL.NUM_LAYERS)
            ]
        )

        self.final_norm = nn.LayerNorm(self.hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, batch) -> torch.Tensor:

        num_nodes = batch.x.size(0)
        device = batch.x.device
        edge_index = batch.edge_index
        src, dst = edge_index

        x = self.input_projection(batch.x)
        edge_emb = self.edge_encoder(batch.msg)
        node_last_t = torch.zeros(num_nodes,device=device)

        node_last_t.scatter_reduce_(
            0,
            src,
            batch.t.float(),
            reduce="amax",
            include_self=True)

        t_rel = (batch.t.float()- node_last_t[src])
        time_emb = self.time_encoder(batch.t,t_rel)
        stru_emb = self.gnn(x,edge_index,edge_emb)

        for layer in self.layers:
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_emb=edge_emb,
                time_emb=time_emb,
                edge_t=batch.t.float(),
                history=self.stru_hist)

        x = self.final_norm(x)
        node_idx = torch.arange(num_nodes,device=device)
        current_time = batch.t.float().max().expand(num_nodes)
        self.stru_hist.write(node_idx,stru_emb,current_time)
        return x

    @torch.no_grad()
    def reset_memory(self):
        self.stru_hist.reset()
