import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class XSimGCL (nn.Module):
    def __init__(self, data_generator, args):
        super(XSimGCL, self).__init__()
        self.n_user = data_generator.n_users
        self.n_item = data_generator.n_items
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.args = args
        self.data = data_generator
        self.layers = args.layer_size
        
        self.cl_rate = 0.001
        self.eps = 0.1
        self.temp = 0.5
        print("cl loss rate = {}, eps = {}, temp = {}".format(self.cl_rate, self.eps, self.temp))

    # init the weight of user-item
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_item, self.emb_size)
        
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.norm_adj  = data_generator.norm_adj
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(args.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

    def InfoNCE(self, index, firstView, secondView):
        

        view1 = F.normalize(firstView, dim=1)
        view2 = F.normalize(secondView, dim=1)
        
        view1_embs = view1[index]
        view2_embs = view2[index]

        view1_embs_abs = view1_embs.norm(dim=1)
        view2_embs_abs = view2_embs.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', view1_embs, view2_embs) / torch.einsum('i,j->ij', view1_embs_abs, view2_embs_abs)
        sim_matrix = torch.exp(sim_matrix / self.temp)
        pos_sim = sim_matrix[np.arange(view1_embs.shape[0]), np.arange(view1_embs.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        return loss.mean()
    
    def encoder (self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddings += (F.normalize(torch.rand(ego_embeddings.shape).cuda(), p=2) * torch.sign(ego_embeddings)) * self.eps
                
            all_embeddings.append(ego_embeddings)
            if k == 0:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        ue, ie = torch.split(final_embeddings, [self.n_user, self.n_item])
        aug_ue, aug_ie = torch.split(all_embeddings_cl, [self.n_user, self.n_item])
        return ue, ie, aug_ue, aug_ie
    
    def forward (self, data, is_train):
        
        ue, ie, aug_ue, aug_ie = self.encoder()

        if is_train:
            pos = torch.sum(ue[data[0]] * ie[data[1]], dim=-1)
            neg = torch.sum(ue[data[0]] * ie[data[2]], dim=-1)
            loss = torch.mean(F.softplus(neg-pos))
            loss += self.cl_rate * self.InfoNCE(data[0], ue, aug_ue)
            loss += self.cl_rate * self.InfoNCE(data[1], ie, aug_ie)
            return loss
        else:
            return torch.sum(ue[data[0]] * ie[data[1]], dim=-1)
        
