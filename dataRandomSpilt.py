import math
import torch
import configSetting


def dataSplitChoice(data, fold, s, val_ratio=configSetting.val_ratio, test_ratio=configSetting.test_ratio):
    if configSetting.choice:
        torch.manual_seed(s)
        num_nodes = data.num_nodes
        row_original, col_original = data.edge_index
        data.edge_index = None
        n_v = int(math.floor(val_ratio * row_original.size(0)))
        n_t = int(math.floor(test_ratio * row_original.size(0)))
        n_a = int(math.floor(row_original.size(0)))

        perm = torch.randperm(row_original.size(0))
        start_step = int(fold / configSetting.fold * perm.size().numel())
        perm_repeat = torch.cat([perm, perm], dim=0)
        row, col = row_original[perm_repeat], col_original[perm_repeat]
        r, c = row[start_step:start_step + n_v], col[start_step:start_step + n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[start_step + n_v:start_step + n_v + n_t], col[start_step + n_v:start_step + n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[start_step + n_v + n_t:start_step + n_a], col[start_step + n_v + n_t:start_step + n_a]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)

        #edge_type
        relation = data.edge_type[perm_repeat]
        data.train_pos_edge_type = relation[start_step + n_v + n_t:start_step + n_a]

        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        diag = torch.diag(neg_adj_mask)
        reshape_diag = torch.diag_embed(diag)
        neg_adj_mask = (neg_adj_mask - reshape_diag).to(torch.bool)
        neg_adj_mask[row_original, col_original] = 0
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask
        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)
        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)
        return data

    elif configSetting.choice != True:
        torch.manual_seed(s)
        num_nodes = data.num_nodes
        row_original, col_original = data.edge_index
        data.edge_index = None
        n_v = int(math.floor(val_ratio * row_original.size(0)))
        n_t = int(math.floor(test_ratio * row_original.size(0)))
        n_a = int(math.floor(row_original.size(0)))

        perm = torch.randperm(row_original.size(0))
        start_step = int(fold / configSetting.fold * perm.size().numel())
        perm_repeat = torch.cat([perm, perm], dim=0)
        row, col = row_original[perm_repeat], col_original[perm_repeat]
        r, c = row[start_step:start_step + n_v], col[start_step:start_step + n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        data.val_neg_edge_index = torch.stack([c, r], dim=0)
        r, c = row[start_step + n_v:start_step + n_v + n_t], col[start_step + n_v:start_step + n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        data.test_neg_edge_index = torch.stack([c, r], dim=0)
        r, c = row[start_step + n_v + n_t:start_step + n_a], col[start_step + n_v + n_t:start_step + n_a]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)

        #edge_type
        relation = data.edge_type[perm_repeat]
        data.train_pos_edge_type = relation[start_step + n_v + n_t:start_step + n_a]

        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        diag = torch.diag(neg_adj_mask)
        reshape_diag = torch.diag_embed(diag)
        neg_adj_mask = (neg_adj_mask - reshape_diag).to(torch.bool)
        neg_adj_mask[row_original, col_original] = 0
        neg_adj_mask[col_original[:n_v + n_t], row_original[:n_v + n_t]] = 0
        data.train_neg_adj_mask = neg_adj_mask
        return data





