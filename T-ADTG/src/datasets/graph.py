from raphtory import Graph, algorithms
from .temporal_data import TemporalData

from .loader import EventLoader


# Create Raphtory Temporal graph 

def create_graph(data: TemporalData):
    g = Graph()
    
    for event in data :
        g.add_edge(event.t.item(), event.src.item(), event.dst.item())
    
    return g  


## Negative Pairs Sampling Based on Temporal Graph Clustering : Lauvain

def negative_samples(g: Graph, node_id, nodes):
    
    clustering = algorithms.louvain(g)

    node_cluster = clustering[node_id]

    return [n for n in nodes if clustering[n] != node_cluster]




