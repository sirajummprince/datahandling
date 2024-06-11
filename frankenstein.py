import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip

raw_dir = 'tmp/raw/'

path = os.path.join(raw_dir, 'FRANKENSTEIN.node_attrs')
node_attrs = pd.read_csv(path, sep = ',', header = None)
node_attrs.index += 1

path = os.path.join(raw_dir, 'FRANKENSTEIN.edges')
edge_index = pd.read_csv(path, sep = ',', names = ['source', 'target'])
edge_index.index += 1

path = os.path.join(raw_dir, 'FRANKENSTEIN.graph_idx')
graph_idx = pd.read_csv(path, sep=',', names=['idx'])
graph_idx.index += 1

path = os.path.join(raw_dir, 'FRANKENSTEIN.graph_labels')
graph_labels = pd.read_csv(path, sep=',', names=['label'])
graph_labels.index += 1

g_idx = 2345

node_ids = graph_idx.loc[graph_idx['idx'] == g_idx].index

attributes = node_attrs.loc[node_ids, :]

edges = edge_index.loc[edge_index['source'].isin(node_ids)]
edge_ids = edges.index

label = graph_labels.loc[g_idx]

print('Nodes:', node_ids.shape)
print('Attributes:', attributes.shape)
print('Edges:', edges.shape)
print('Label:', label.shape)

print('Nodes:', node_ids)
print('Attributes:', attributes)
print('Edges:', edges)
print('Label:', label)

edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype = torch.long)
map_dict = {v.item():i for i,v in enumerate(torch.unique(edge_idx))}
map_edge = torch.zeros_like(edge_idx)

for k,v in map_dict.items():
    map_edge[edge_idx == k] = v

print('Map Dictionary:', map_dict)
print('Map Edge Shape:', map_edge.shape)
print('Map Edge:', map_edge)

attrs = torch.tensor(attributes.to_numpy(), dtype = torch.float)
pad = torch.zeros((attrs.shape[0], 4), dtype = torch.float)
x = torch.cat((attrs, pad), dim=-1)

edge_idx = map_edge.long()

np_lab = label.to_numpy()
y = torch.tensor(np_lab if np_lab[0] == 1 else [0], dtype = torch.long)

print('X Shape:', x.shape)

graph = Data(x = x, edge_index = edge_idx,  y = y)

print('Graph:', graph)