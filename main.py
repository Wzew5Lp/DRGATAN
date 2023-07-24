import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
# from torch_geometric.nn import RGATConv,GATConv
from modelLayers import RGATConv
from modelLayers.RelationAware import GAttNet,GATConv;
from torch_geometric.data import Data
import random
import copy
import configSetting
from GraphAutoEnconder import GAE

from dataRandomSpilt import dataSplitChoice
#
# import os
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# edge_list = pd.read_csv('./deepddi-dataset/deepddi_edge_list.csv', header=None, sep=' ')
# # print(edge_list)
# # print(torch.tensor(edge_list.to_numpy(),dtype=torch.long).t().contiguous())
# edge_index = torch.tensor(edge_list.to_numpy(), dtype=torch.long).t().contiguous()
# # print(edge_index)
#
# edge_type = pd.read_csv('./deepddi-dataset/deepddi_edge_type.csv', header=None)
# # print(edge_type.to_numpy().flatten())
# edge_type_0 = [etid-1 for etid in edge_type.to_numpy().flatten()]
# # print(edge_type_0)
# # edge_type = torch.tensor(edge_type.to_numpy().flatten(), dtype=torch.long)
# edge_type = torch.tensor(edge_type_0, dtype=torch.long)
#
#
# enti = pd.read_csv('./deepddi-dataset/deepddi-drug-features.csv', sep=' ', header=None, index_col=0)
# # print(enti)
# # features = torch.tensor(enti.loc['DB00006':'DB13925','DB00006':'DB13925'].values, dtype=torch.float)
# features = torch.tensor(enti.values,dtype=torch.float)
# # print(features)
# num_features = features.size(1)
# # print(num_features)
#
# num_nodes = len(set(edge_list.to_numpy().flatten()))
# # print(num_nodes)
#
# num_relations = len(set(edge_type.tolist()))
# # print(num_relations)
#
# # x = torch.randn(num_nodes, 512)
#
# fea_pca = pd.read_csv('./deepddi-dataset/deepddi_feature_pca_100.csv', sep=',', header=None)
# print(fea_pca.values)
# print(fea_pca.values.shape)
# featuresPCA = torch.tensor(fea_pca.values,dtype=torch.float)
# pca_num_features = featuresPCA.size(1)

"""
my_dataset_test
"""
my_edge_list = pd.read_csv('./my_dataset/my_edge_list.csv', header=None, sep='\t')
my_edge_index = torch.tensor(my_edge_list.to_numpy(), dtype=torch.long).t().contiguous()

my_edge_type = pd.read_csv('./my_dataset/my_edge_type.csv', header=None)
my_edge_type = torch.tensor(my_edge_type.to_numpy().flatten(), dtype=torch.long)

my_num_relations = len(set(my_edge_type.tolist()))
my_num_nodes = len(set(my_edge_list.to_numpy().flatten()))


myfea = pd.read_csv('./my_dataset/my_drug_features.csv',sep = ' ',header= None).values
my_features = torch.tensor(myfea,dtype = torch.float)
my_num_features = my_features.size(1)





class DRGATAN(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DRGATAN, self).__init__()

        """
            attention_mechanism:{across-relation , within-relation}
        """
        self.conv1 = RGATConv(in_channels, 4 * out_channels, num_relations=my_num_relations,
                             heads=2, concat=False, add_self_loops=False, attention_mechanism='across-relation')
        self.convS = RGATConv(4 * out_channels, out_channels + configSetting.beta_on, num_relations=my_num_relations,
                             heads=2, concat=False, add_self_loops=False, attention_mechanism='across-relation')

        self.conv2 = RGATConv(in_channels, 4 * out_channels, num_relations=my_num_relations,
                             heads=2, concat=False, add_self_loops=False, attention_mechanism='across-relation')
        self.convT = RGATConv(4 * out_channels, out_channels + configSetting.beta_on, num_relations=my_num_relations,
                             heads=2, concat=False, add_self_loops=False, attention_mechanism='across-relation')

        """
        (w/o) relation
        """
        # self.conv1 = GATConv(in_channels, out_channels + config.beta_on,
        #                      heads=4, concat=False, add_self_loops=False)
        # self.conv2 = GATConv(in_channels, out_channels + config.beta_on,
        #                      heads=4, concat=False, add_self_loops=False)

        """
        (w/o) relation aware
        """

        self.ra1 = GAttNet(configSetting.dir,configSetting.label_num,in_channels,out_channels*4)
        self.ra2 = GAttNet(configSetting.dir,configSetting.label_num,out_channels*4,out_channels)
        # self.ra1 = GATConv(in_channels, out_channels,
        #                      heads=32, concat=False, add_self_loops=False)
        # self.ra2 = GATConv(4*out_channels, out_channels,
        #                   heads=16, concat=False, add_self_loops=False)






    def forward(self, x, edge_index, edge_type):
        F.dropout(x, p=0.6, training=self.training)
        x_s = F.elu(self.conv1(x, edge_index, edge_type))
        x_in = F.elu(self.convS(x_s, edge_index, edge_type))
        x_t = F.elu(self.conv2(x, edge_index, edge_type))
        x_out = F.elu((self.convT(x_t, edge_index, edge_type)))
        x_self = F.elu(self.ra1(x, edge_index))
        return x_in, x_out, x_self


def train():
    model.train()
    optimizer.zero_grad()
    z_in, z_out, z_self = model.encode(data.x, data.train_pos_edge_index, data.train_pos_edge_type)
    loss = model.recon_loss(z_in, z_out, z_self, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self = model.encode(data.x, data.train_pos_edge_index, data.train_pos_edge_type)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def testfinal(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self = model.encode(data.x, data.train_pos_edge_index, data.train_pos_edge_type)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def initialize_list():
    lists = [[] for _ in range(6)]
    return [lists[i] for i in range(6)]

if __name__ == '__main__':
    target = ["auc", "ap", "acc", "f1", "pre", "re"]
    auc_list, ap_list, f1_list, acc_list, pre_list, re_list = initialize_list()
    target_list = [auc_list, ap_list,  f1_list, acc_list, pre_list, re_list]
    for i in range(configSetting.number):


        auc_l, ap_l, f1_l, acc_l, pre_l, re_l = initialize_list()
        target_l = [auc_l, ap_l, f1_l, acc_l, pre_l, re_l]
        for fold in range(configSetting.fold):
            data = Data(edge_index=my_edge_index, x=my_features, edge_type=my_edge_type)
            data = dataSplitChoice(data,fold,0)
            print(data)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data.to(device)
            model = GAE(DRGATAN(data.num_features,configSetting.out_channels)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=configSetting.lr)

            min_loss_val = configSetting.min_loss_val
            best_model = None
            min_epoch = configSetting.min_epoch
            for epoch in range(1, configSetting.epochs + 1):
                loss = train()
                if epoch % 10 == 0:
                    auc, ap, acc, f1, pre, re = test(data.val_pos_edge_index, data.val_neg_edge_index)
                    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f},'
                          .format(epoch, auc, ap, acc, f1, pre, re))
                if epoch > min_epoch and loss <= min_loss_val:
                    min_loss_val = loss
                    best_model = copy.deepcopy(model)
                    torch.save({'model': model.state_dict()}, './saveModel/Dblind_DRGATDDI_best.pth')
            model = best_model
            auc, ap, acc, f1, pre, re = testfinal(data.test_pos_edge_index, data.test_neg_edge_index)
            print('final. AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f},'
                  .format(auc, ap, acc, f1, pre, re))
            for j in range(6):
                target_l[j].append(eval(target[j]))
        for j in range(6):
            target_list[j].append(np.mean(target_l[j]))
    for j in range(6):
        print(np.mean(target_list[j]), np.std(target_list[j]))

