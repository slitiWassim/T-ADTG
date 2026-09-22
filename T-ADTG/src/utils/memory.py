import torch
from torch import nn
from torch_geometric.nn import Linear
import torch.nn.functional as F


class Memory(nn.Module):
    """Per-node memory state, updated by a recurrent cell (RNN/GRU/LSTM).

    Only the most recent state per node is ever needed, so memory is a
    single [num_nodes, dimension] tensor rather than a windowed queue.
    """
    def __init__(self, num_nodes, config, device='cpu'):
        super().__init__()
        self.num_nodes     = num_nodes
        self.dimension     = config.MODEL.HIDDEN_CHANNELS
        self.device        = device


        self.recurrent_network = nn.GRUCell(input_size=self.dimension, hidden_size=self.dimension)
        self.lin = nn.Sequential(
            Linear(-1, self.dimension),
            nn.ELU(),
            Linear(self.dimension, self.dimension),
        )

        self.init_memory()

    def init_memory(self):
        """Resets every node's memory to zero. Call at the start of each
        epoch so memory doesn't leak state across epochs."""
        self.memory = torch.zeros(self.num_nodes, self.dimension, device=self.device)

    def get_memory(self, node_idxs):
        return self.memory[node_idxs].detach()

    def set_memory(self, node_idxs, values):
        self.memory = self.memory.detach().index_put((node_idxs,), values)

    def forward(self, x, idx):
        x = self.lin(x)
        x = F.normalize(x, dim=-1)
        mem = self.get_memory(idx)             
        out = self.recurrent_network(x, mem) 
        self.set_memory(idx, out)             

        return out, mem