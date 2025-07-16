import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDrop(nn.Module):
    """ Drop edges in a graph.
    """
    def __init__(self, resize_val=False):
        super(EdgeDrop, self).__init__()
        self.resize_val = resize_val

    def forward(self, adj, keep_rate):
        """
        :param adj: torch_adj in data_handler
        :param keep_rate: ratio of preserved edges
        :return: adjacency matrix after dropping edges
        """
        if keep_rate == 1.0: return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask] / (keep_rate if self.resize_val else 1.0)
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class CCDR(nn.Module):
    def __init__(self, ui_data_a, ui_data_b, args):
        super(CCDR, self).__init__()

        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.emb_dim = self.args.emb_size
        
        self.n_users_a = ui_data_a.n_users
        self.n_users_b = ui_data_b.n_users
        self.n_items_a = ui_data_a.n_items
        self.n_items_b = ui_data_b.n_items
        self.n_layers = args.layer_size
        
        self.uiMat_a = self._convert_sp_mat_to_sp_tensor(ui_data_a.norm_adj).to(args.device)
        self.uiMat_b = self._convert_sp_mat_to_sp_tensor(ui_data_b.norm_adj).to(args.device)

        self.uA = nn.Embedding(self.n_users_a, self.emb_dim)
        self.iA = nn.Embedding(self.n_items_a, self.emb_dim)
        self.uB = nn.Embedding(self.n_users_b, self.emb_dim)
        self.iB = nn.Embedding(self.n_items_b, self.emb_dim)

        self.edge_drop = EdgeDrop()

        self.inter_cl = 0.001
        self.cl_a = 0.1 
        self.cl_b = 0.1

       
        def init_weights(m):
            if isinstance(m, nn.Linear):
                print(m)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            if isinstance(m, nn.Embedding):
                print(m)
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        overlap = np.intersect1d(ui_data_a.full_data["reviewerID"].unique(), ui_data_b.full_data["reviewerID"].unique())
        self.overlap_users = np.array([i for i in range (0, len(overlap))])
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()
      
    def graph_embedding(self, is_a, mat):
        if is_a:
            ego = torch.cat([self.uA.weight, self.iA.weight], 0)
            n_items = self.n_items_a
            n_users = self.n_users_a
        else:
            ego = torch.cat([self.uB.weight, self.iB.weight], 0)
            n_items = self.n_items_b
            n_users = self.n_users_b
        
        embs = [ego]
        for k in range (self.args.layer_size):
            ego = torch.sparse.mm(mat, ego)
            embs.append(ego)
        
        embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        ue, ie = torch.split(embs, [n_users, n_items])
        return ue, ie
    


    

    def cal_infonce_loss(self, view1, view2, index):
        if len(index) ==0:
            view1_embs = F.normalize(view1, dim=1)
            view2_embs = F.normalize(view2, dim=1)
        else:
            index = torch.unique(torch.Tensor(index).type(torch.long)).to(self.args.device)

            view1 = F.normalize(view1, dim=1)
            view2 = F.normalize(view2, dim=1)
            
            view1_embs = view1[index]
            view2_embs = view2[index]

        view1_embs_abs = view1_embs.norm(dim=1)
        view2_embs_abs = view2_embs.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', view1_embs, view2_embs) / torch.einsum('i,j->ij', view1_embs_abs, view2_embs_abs)
        sim_matrix = torch.exp(sim_matrix / 0.7)
        pos_sim = sim_matrix[np.arange(view1_embs.shape[0]), np.arange(view1_embs.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        return loss.mean()
    
    def forward(self, data_a, data_b, phase):
        ua, ia = self.graph_embedding(True, self.uiMat_a)
        ub, ib = self.graph_embedding(False, self.uiMat_b)

        
        if phase == "test":
            scores_a = torch.sum(ua[data_a[0]] * ia[data_a[1]], dim=-1)
            scores_b = torch.sum(ub[data_b[0]] * ib[data_b[1]], dim=-1)
            return scores_a, scores_b
        
        if phase == "train-join":
            pos_a = torch.sum(ua[data_a[0]] * ia[data_a[1]], dim=-1)
            neg_a = torch.sum(ua[data_a[0]] * ia[data_a[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_a - pos_a))

            pos_b = torch.sum(ub[data_b[0]] * ib[data_b[1]], dim=-1)
            neg_b = torch.sum(ub[data_b[0]] * ib[data_b[2]], dim=-1)
            loss += torch.mean(F.softplus(neg_b - pos_b))

            ## intra-domain
            aug_a1 = self.edge_drop(self.uiMat_a, 0.7)
            aug_a2 = self.edge_drop(self.uiMat_a, 0.7)
            ua_v1, ia_v1 = self.graph_embedding(True, aug_a1)
            ua_v2, ia_v2 = self.graph_embedding(True, aug_a2)
            loss += self.cl_a * (
                self.cal_infonce_loss(ua_v1, ua_v2, data_a[0]) + self.cal_infonce_loss(ia_v1, ia_v2, data_a[1])
            )

            aug_b1 = self.edge_drop(self.uiMat_b, 0.7)
            aug_b2 = self.edge_drop(self.uiMat_b, 0.7)
            ub_v1, ib_v1 = self.graph_embedding(False, aug_b1)
            ub_v2, ib_v2 = self.graph_embedding(False, aug_b2)
            loss += self.cl_b * (
                self.cal_infonce_loss(ub_v1, ub_v2, data_b[0]) + self.cal_infonce_loss(ib_v1, ib_v2, data_b[1])
            )

            ## inter-domain
            oua, oub = ua[self.overlap_users], ub[self.overlap_users]
            loss += self.inter_cl * self.cal_infonce_loss(oua, oub, [])
            return loss
        
        if phase == "train-a":
            pos_a = torch.sum(ua[data_a[0]] * ia[data_a[1]], dim=-1)
            neg_a = torch.sum(ua[data_a[0]] * ia[data_a[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_a - pos_a))

            ## intra-domain
            aug_a1 = self.edge_drop(self.uiMat_a, 0.7)
            aug_a2 = self.edge_drop(self.uiMat_a, 0.7)
            ua_v1, ia_v1 = self.graph_embedding(True, aug_a1)
            ua_v2, ia_v2 = self.graph_embedding(True, aug_a2)
            loss += self.cl_a * (
                self.cal_infonce_loss(ua_v1, ua_v2, data_a[0]) + self.cal_infonce_loss(ia_v1, ia_v2, data_a[1])
            )

        
            ## inter-domain
            oua, oub = ua[self.overlap_users], ub[self.overlap_users]
            loss += self.inter_cl * self.cal_infonce_loss(oua, oub, [])
            return loss

        if phase == "train-b":

            pos_b = torch.sum(ub[data_b[0]] * ib[data_b[1]], dim=-1)
            neg_b = torch.sum(ub[data_b[0]] * ib[data_b[2]], dim=-1)
            loss = torch.mean(F.softplus(neg_b - pos_b))

            ## intra-domain

            aug_b1 = self.edge_drop(self.uiMat_b, 0.7)
            aug_b2 = self.edge_drop(self.uiMat_b, 0.7)
            ub_v1, ib_v1 = self.graph_embedding(False, aug_b1)
            ub_v2, ib_v2 = self.graph_embedding(False, aug_b2)
            loss += self.cl_b * (
                self.cal_infonce_loss(ub_v1, ub_v2, data_b[0]) + self.cal_infonce_loss(ib_v1, ib_v2, data_b[1])
            )

            ## inter-domain
            oua, oub = ua[self.overlap_users], ub[self.overlap_users]
            loss += self.inter_cl * self.cal_infonce_loss(oua, oub, [])
            return loss
        



            

