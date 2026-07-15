
from collections import deque

import torch
from torch import nn
from torch_geometric.nn import Linear    
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, num_nodes, num_timeslots, dimension, device='cpu',
                 history_retrieve="last", recurrent="gru" , normalize = True , out_normalize= False):
        super().__init__()
        self.num_nodes         = num_nodes
        self.history_dimension = dimension
        self.num_timeslots     = num_timeslots
        self.history_retrieve  = history_retrieve
        self.recurrent         = recurrent
        self.device            = device
        self.normalize         = normalize
        self.out_normalize     = out_normalize

        if recurrent == "rnn":
            self.recurrent_network = nn.RNNCell(input_size=dimension, hidden_size=dimension)
        elif recurrent == "gru":
            self.recurrent_network = nn.GRUCell(input_size=dimension, hidden_size=dimension)
        elif recurrent == "lstm":
            self.recurrent_network = nn.LSTMCell(input_size=dimension, hidden_size=dimension)
        
        

        self.history = deque(maxlen=num_timeslots)
        for _ in range(num_timeslots):
            self.history.append(
                torch.zeros(num_nodes, dimension, requires_grad=False).to(device)
            )

        self.lin = nn.Sequential(
            Linear(-1, dimension),
            nn.ELU(),
            Linear(dimension, dimension),
        )

    ### Difference : Give you the option to return the mean of the historical hidden state of 
    ### the node representation over time produced by the GRU layer     

    def get_history(self, node_idxs):
        # always detached — used as GRU hidden state only, never for loss
        if self.history_retrieve == "last":
            return self.history[-1][node_idxs].detach()
        elif self.history_retrieve == "mean":
            return torch.stack(
                [m[node_idxs] for m in self.history], dim=0
            ).mean(dim=0).detach()

    def set_history(self, node_idxs, values):
        # clone the current last slot (detached) to use as new base
        new_last = self.history[-1].detach().clone()
        
        # shift: slot[i] = slot[i+1] for all but last, all detached
        shifted = [self.history[i+1].detach().clone() 
                   for i in range(len(self.history) - 1)]
        
        # rebuild the deque cleanly
        self.history.clear()
        for slot in shifted:
            self.history.append(slot)
        
        # last slot: non-inplace index_put preserves grad_fn from values
        new_last = new_last.index_put((node_idxs,), values)  # ← live grad ✓
        self.history.append(new_last)

    def forward(self, x, idx, update=True):
        x   = self.lin(x)
        ## Normalize 
        if self.normalize :
            x = F.normalize(x, dim=-1)

        mem = self.get_history(idx)              # detached hidden state
        if self.recurrent == "lstm":
            out = self.recurrent_network(x, (mem, torch.zeros_like(mem)))[0]
        else:
            out = self.recurrent_network(x, mem) # out has live grad_fn ✓
        
        if self.out_normalize :
            out = F.normalize(out,dim=-1)
        
        if update:
            self.set_history(idx, out)           # history[-1] keeps grad_fn ✓
        
        return out, mem
