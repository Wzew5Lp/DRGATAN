import torch
import pandas as pd
import numpy as np







my_edge_list = pd.read_csv('./my_edge_list.csv', header=None, sep='\t')
my_edge_index = torch.tensor(my_edge_list.to_numpy(), dtype=torch.long).t().contiguous()

case_ratio = int(my_edge_index.size(1)*0.95)


train_edge_index = my_edge_index[:,0:int(my_edge_index.size(1)*0.95)]
case_edge_index = my_edge_index[:,case_ratio:case_ratio+int(my_edge_index.size(1)*0.05)+1]

train_nodes = len(set(train_edge_index.numpy().flatten()))

print(train_nodes)
print(my_edge_index)
print(my_edge_index.size())
print(train_edge_index)
print(train_edge_index.size())
print(case_edge_index)
print(case_edge_index.size())


my_edge_type = pd.read_csv('./my_edge_type.csv', header=None)
my_edge_type = torch.tensor(my_edge_type.to_numpy().flatten(), dtype=torch.long)
train_edge_type = my_edge_type[0:int(my_edge_type.size(0)*0.95)]
print(my_edge_type.size())
print(my_edge_type[0:int(my_edge_type.size(0)*0.95)].size())
print(len(set(train_edge_type.tolist())))

#
# my_num_relations = len(set(my_edge_type.tolist()))
# my_num_nodes = len(set(my_edge_list.to_numpy().flatten()))
#
# my_features = torch.randn(my_num_nodes, 100)
# my_num_features = my_features.size(1)
