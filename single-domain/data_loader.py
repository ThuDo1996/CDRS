import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import json
from tqdm import tqdm
import warnings

class Data(object):
    def __init__(self, path, args):
        super(Data, self).__init__()
        domain = args.domain
        
        ### Load data
        self.train = pd.read_csv(path + "train_"+domain+".csv")
        valid = pd.read_csv(path + "valid_"+domain+".csv")
        test = pd.read_csv(path + "test_"+domain+".csv")
        full_data = pd.concat([self.train, valid, test])

        ### Data statistic
        self.args = args
        self.n_users = max(full_data["uid"]) + 1
        self.n_items = max(full_data['iid']) + 1
        self.n_train = self.train.shape[0]
        print("Number of users = {},  items = {}, ratings = {}".format(self.n_users, self.n_items, self.n_train))

        ### Create interaction matrix
        print("Create interaction matrix")
        self.norm_adj = self.load_graph()
            
        ### Create evaluation data
        print("Create evaluation data")
        self.valid = self.create_test_ranking(full_data, valid)
        self.test = self.create_test_ranking(full_data, test)    

        

    ### split items into popular and unpopular for visualization
    def split_popular_rare_items(self):
      data = self.train
      num_items = len(data["iid"].unique())
      num_popular = int(num_items*0.05)
      num_rare = int(num_items*0.05)
      x = data['iid'].value_counts()
      popular = x.head(num_popular).keys().tolist()[0:1000]
      rare = x.tail(num_rare).keys().tolist()[0:1000]
      return np.array(popular), np.array(rare)
    
    #### Create training data:
    def construct_new_train(self):
      item_list = [i for i in range (self.n_items)]
      data = self.train

      new_data = []
      for user in data["uid"].unique():
        user_data = data.loc[data["uid"]==user]
        interacted_items  = user_data["iid"].unique() 
        neg_items = list(set(item_list) - set(interacted_items))
        if len(interacted_items) > len(neg_items):
          selected_neg_items = random.choices(list(neg_items), k=len(interacted_items))  # with replacement
        else:
          selected_neg_items = random.sample(list(neg_items), len(interacted_items))  # without replacement
        generated_data = pd.DataFrame({"uid": np.array([user]*len(selected_neg_items)),"pos_iid":np.array(interacted_items),"neg_iid":np.array(selected_neg_items) }, columns=["uid","pos_iid","neg_iid"] ) 
        new_data.append(generated_data)
      new_train_data = pd.concat(new_data)
      return new_train_data.sample(frac=1)

    def get_data(self,batchsize, is_triplet=True):
      if is_triplet:
        data = self.construct_new_train()
        uid = torch.tensor(data["uid"].values, dtype=torch.long)
        pos_iid = torch.tensor(data["pos_iid"].values, dtype=torch.long)
        neg_iid = torch.tensor(data["neg_iid"].values, dtype=torch.long)
        dataset = TensorDataset(uid, pos_iid, neg_iid)
        
      else:
        data = self.train
        uid = torch.tensor(data["uid"].values, dtype=torch.long)
        pos_iid = torch.tensor(data["iid"].values, dtype=torch.long)
        dataset = TensorDataset(uid, pos_iid)
      data_iter = DataLoader(dataset, batchsize, shuffle=True)
      return data_iter
    

    ### Create interaction graph and normalization
    def construct_interactions_matrix (self,n_users, n_items, data):
        R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
        for i in range (0, data.shape[0]):
            row =data.iloc[i]
            user, item = int(row["uid"]), int(row["iid"])
            R[user, item] = 1.0
        return R
    
    def load_graph(self):   
      R = self.construct_interactions_matrix(self.n_users, self.n_items, self.train)
      norm_adj = self.create_adj_mat(self.n_users, self.n_items, R)
      return norm_adj
    
    def create_adj_mat(self,n_users,n_items,R_or ):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R_or.tolil()

        adj_mat[:n_users, n_users:] = R
        adj_mat[n_users:, :n_users] = R.T
        adj_mat = adj_mat.todok()


        def normalized_adj_single(adj_mat):
            shape = adj_mat.get_shape()
            rowsum = np.array(adj_mat.sum(1))
            if shape[0] == shape[1]:
              d_inv = np.power(rowsum, -0.5).flatten()
              d_inv[np.isinf(d_inv)] = 0.
              d_mat_inv = sp.diags(d_inv)
              norm_adj_tmp = d_mat_inv.dot(adj_mat)
              norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
            else:
              d_inv = np.power(rowsum, -1).flatten()
              d_inv[np.isinf(d_inv)] = 0.
              d_mat_inv = sp.diags(d_inv)
              norm_adj_mat = d_mat_inv.dot(adj_mat)
            return norm_adj_mat
       

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    
    ### Sampling for evaluation
    def create_test_ranking(self, full_data, test_data):
      uid , candidates, selected_item = [],[],[]
      item_list = [i for i in range (self.n_items)]
      for i in tqdm(range(test_data.shape[0]), ascii=True):
        row = test_data.iloc[i]
        user = row["uid"]
        test_iid = row["iid"]

        all_user_data = full_data.loc[full_data["uid"]==user]
        all_selected_items = all_user_data["iid"].unique()
        neg_items = list(set(item_list) - set(all_selected_items))
        candidate = random.sample(list(neg_items), 99)
        candidate.append(test_iid)

        uid.append(user)
        candidates.append(candidate)
        selected_item.append(test_iid)

      new_data = pd.DataFrame({"uid": np.array(uid),"iid":np.array(selected_item),"candidates":candidates }, columns=["uid","iid","candidates"] )
      return new_data
    

    

      





      


