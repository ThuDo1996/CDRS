import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BiTGCF (nn.Module):
    def __init__(self, ui_data_a, ui_data_b, args):
        super(BiTGCF,self).__init__()
        
        self.args = args
        self.n_user_a = ui_data_a.n_users
        self.n_user_b = ui_data_b.n_users
        self.n_item_a = ui_data_a.n_items
        self.n_item_b = ui_data_b.n_items
        self.emb_size = args.emb_size
        
        self.ui_Mat_a =self._convert_sp_mat_to_sp_tensor(ui_data_a.norm_adj).to(args.device)
        self.ui_Mat_b =self._convert_sp_mat_to_sp_tensor(ui_data_b.norm_adj).to(args.device) 
        
        self.user_embedding_a = nn.Embedding(self.n_user_a, self.emb_size)
        self.user_embedding_b = nn.Embedding(self.n_user_b, self.emb_size)
        self.item_embedding_a = nn.Embedding(self.n_item_a, self.emb_size)
        self.item_embedding_b = nn.Embedding(self.n_item_b, self.emb_size)
        
        overlap = np.intersect1d(ui_data_a.full_data["reviewerID"].unique(), ui_data_b.full_data["reviewerID"].unique())
        self.overlap_users = np.array([i for i in range (0, len(overlap))])
        self.distinct_ua = np.array([i for i in range(len(overlap), ui_data_a.n_users)])
        self.distinct_ub = np.array([i for i in range(len(overlap), ui_data_b.n_users)])
        
        self.lambda_a = 0.7
        self.lambda_b = 0.7

        print("Number of overlap users = {}, distincts in A = {}, distincts in B = {}".format(
            len(self.overlap_users), len(self.distinct_ua), len(self.distinct_ub)
        ))
        def init_weights(m):
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
        self.apply(init_weights)
    

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()
    
    def transfer_layer(self, egoA, egoB):
        ua, ia = torch.split(egoA, [self.n_user_a, self.n_item_a])
        ub, ib = torch.split(egoB, [self.n_user_b, self.n_item_b])

        oua, dua = torch.split(ua, [len(self.overlap_users), self.n_user_a - len(self.overlap_users)])
        oub, dub = torch.split(ub, [len(self.overlap_users), self.n_user_b - len(self.overlap_users)])
        
        laplace_a = self.n_user_a/(self.n_user_a + self.n_user_b)
        laplace_b = self.n_user_b/(self.n_user_a + self.n_user_b)
        
        u_lap = laplace_a * oua + laplace_b * oub
        ua_lam = self.lambda_a * oua + (1 - self.lambda_a) * oub
        ub_lam = self.lambda_b * oub + (1 - self.lambda_b) * oua
        
        new_oua = (u_lap + ua_lam)/2
        new_oub = (u_lap + ub_lam)/2
        
        egoA = torch.cat([new_oua, dua, ia], dim=0)
        egoB = torch.cat([new_oub, dub, ib], dim=0)
        
        return egoA, egoB
    
    def forward(self, data_a, data_b, phase):
        def one_graph_layer_gcf (A_hat, ego_embeddings,k):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            ego_embeddings = side_embeddings + torch.multiply(ego_embeddings, side_embeddings)
            return ego_embeddings
        
        egoA = torch.cat([self.user_embedding_a.weight, self.item_embedding_a.weight],0)
        egoB = torch.cat([self.user_embedding_b.weight, self.item_embedding_b.weight],0)

        embsA, embsB = [egoA], [egoB]
        
        for k in range (self.args.layer_size):
            egoA = one_graph_layer_gcf(self.ui_Mat_a, egoA, k)
            egoB = one_graph_layer_gcf(self.ui_Mat_b, egoB, k)
            
            egoA, egoB = self.transfer_layer(egoA, egoB)
            
            embsA.append(egoA)
            embsB.append(egoB)
        
        embsA = torch.mean(torch.stack(embsA, dim=1), dim=1)
        embsB = torch.mean(torch.stack(embsB, dim=1), dim=1)
        
        ua, ia = torch.split(embsA, [self.n_user_a, self.n_item_a])
        ub, ib = torch.split(embsB, [self.n_user_b, self.n_item_b])
        
        if phase == "test":
            score_a = torch.sum(ua[data_a[0]] * ia[data_a[1]], dim=-1)
            score_b = torch.sum(ub[data_b[0]] * ib[data_b[1]], dim=-1)
            return score_a, score_b
        
        if phase =="train-join":
            pos_a = torch.sum(ua[data_a[0]] * ia[data_a[1]], dim=-1)
            neg_a = torch.sum(ua[data_a[0]] * ia[data_a[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_a - pos_a))
            
            pos_b = torch.sum(ub[data_b[0]] * ib[data_b[1]], dim=-1)
            neg_b = torch.sum(ub[data_b[0]] * ib[data_b[2]], dim=-1)
            loss += torch.mean(F.softplus(neg_b - pos_b))
            return loss
        
        if phase =="train-a":
            pos_a = torch.sum(ua[data_a[0]] * ia[data_a[1]], dim=-1)
            neg_a = torch.sum(ua[data_a[0]] * ia[data_a[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_a - pos_a))
            return loss
        
        if phase =="train-b":
            pos_b = torch.sum(ub[data_b[0]] * ib[data_b[1]], dim=-1)
            neg_b = torch.sum(ub[data_b[0]] * ib[data_b[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_b - pos_b))
            return loss
            
            