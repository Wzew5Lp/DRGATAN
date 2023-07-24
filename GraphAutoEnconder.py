import torch

from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, f1_score, recall_score, precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

from torch_geometric.utils import degree
import torch.nn.functional as F

from torch_geometric.nn.inits import reset

import configSetting


class Predictor(torch.nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.align_in = torch.nn.Linear(configSetting.out_channels, configSetting.out_channels)
        self.align_out = torch.nn.Linear(configSetting.out_channels, configSetting.out_channels)
        self.dnn = torch.nn.Sequential(
                    torch.nn.Linear(64, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 1),
        )

    def forward(self, z_in, z_out, z_self, edge_index, sigmoid=True):

        tensor_degreeout = degree(edge_index[0], dtype=torch.long)

        tensor_degreeout = tensor_degreeout / (len(set(edge_index.flatten().tolist())) - 1)


        tensor_degreeout = F.normalize(tensor_degreeout, dim=0)

        real_degreeout = tensor_degreeout[edge_index[0]]


        tensor_degreein = degree(edge_index[1],dtype=torch.long)
        tensor_degreein = tensor_degreein / (len(set(edge_index.flatten().tolist())) - 1)
        tensor_degreein = F.normalize(tensor_degreein, dim=0)
        real_degreein = tensor_degreein[edge_index[1]]

        if configSetting.beta_on:


            value = ((real_degreeout+z_out[edge_index[0]][:, -1]) * configSetting.beta +
                     (z_out[edge_index[0]][:, :-1] * self.align_out(z_self[edge_index[1]])).sum(dim=1) * configSetting.alpha +
                     (real_degreein+z_in[edge_index[1]][:, -1]) * configSetting.beta +
                     (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]][:, :-1]).sum(dim=1) * configSetting.alpha)



        else:

            """
            (w/o) degree cent
            """

            value = ((z_out[edge_index[0]] * self.align_out(z_self[edge_index[1]])).sum(dim=1)*0.5 +
                     (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]]).sum(dim=1)*0.5)


        return torch.sigmoid(value) if sigmoid else value


class GAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = Predictor() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):

        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):

        return self.decoder(*args, **kwargs)

    def recon_loss(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index=None):

        pos_loss = -torch.log(self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True) + configSetting.EPS).mean()

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z_self.size(0))
        neg_loss = -torch.log(1 - self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True) + configSetting.EPS).mean()
        return pos_loss + neg_loss

    def test(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index):

        pos_y = z_self.new_ones(pos_edge_index.size(1))
        neg_y = z_self.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred), accuracy_score(y, pred.round()),f1_score(y, pred.round()), precision_score(y, pred.round()), recall_score(y, pred.round())
