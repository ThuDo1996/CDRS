import torch
import torch.optim as optim
import numpy as np
from data_loader import Data
import torch.optim as optim
from tqdm import tqdm
import argparse
from XSimGCL import XSimGCL
from DirectAU import DirectAU

def test_hit_ratio(user_data, emb_model):
      def get_top_N_items(candidate_scores, candidates):
        top_n  = 20
        candidate_scores = candidate_scores.detach().cpu().numpy()
        top_indices = np.argsort(candidate_scores)[-top_n:][::-1]
        top_candidates = [candidates[i] for i in top_indices]
        return top_candidates
 
      hit_5, hit_10 =0,0
      count, start =0,0
      ndcg_5, ndcg_10 = 0,0
      users, items = [],[]
      
      for i in range(user_data.shape[0]):
        row = user_data.iloc[i]
        
        item_candidate = row["candidates"]
        uid = np.full(len(item_candidate), int(row["uid"]))

        users.append(uid)
        items.append(item_candidate)

      users = np.concatenate(users) if users else np.array([])
      items = np.concatenate(items) if items else np.array([])
      scores = emb_model([users, items], False)
        
      for i in tqdm(range(user_data.shape[0]), ascii=True):
        row = user_data.iloc[i]
        true_item = row["iid"]
        candidate = row["candidates"]

        predictions = scores[start: start + len(candidate)]
        top_k_rec = get_top_N_items(predictions, candidate)
        index = np.where(top_k_rec == true_item)[0]
        
        if index.size > 0:
          index = index[0]
          if index < 5:
              hit_5 += 1
              ndcg_5 += 1 / np.log2(index + 2)
          if index < 10:
              hit_10 += 1
              ndcg_10 += 1 / np.log2(index + 2)

        count+=1 
        start+=len(item_candidate)
      return hit_5/count, ndcg_5/count, hit_10/count, ndcg_10/count

def train(args):
    path = args.data_path
    data_generator = Data(path, args)
    if args.model == "DirectAU":
      print("Training with DirectAU model")
      emb_model = DirectAU(data_generator, args).to(args.device)
    if args.model == "XSimGCL":
      print("Training with XSimGCL model")
      emb_model = XSimGCL(data_generator, args).to(args.device)
      
    optimizer_emb= optim.Adam(emb_model.parameters(), lr = args.lr, weight_decay = args.wd)
          
    best_hit = -1
    count_stop = 0
    index = 0
    
    while count_stop < 7 and index <100:
        emb_model.train()
        print("-------------- Loop {}----------------".format(index))
        if args.model =="DirectAU":
          train_data = data_generator.get_data(args.batch_size, False)
          for u, pos_i in tqdm(train_data):
            loss = emb_model([u, pos_i], True)
            
            optimizer_emb.zero_grad()
            loss.backward()
            optimizer_emb.step()
        else:
          train_data = data_generator.get_data(args.batch_size)
          for u, pos_i, neg_i in tqdm(train_data):
            loss = emb_model([u, pos_i, neg_i], True)
            
            optimizer_emb.zero_grad()
            loss.backward()
            optimizer_emb.step()
        
      
        emb_model.eval()
        _,_, hit, ndcg = test_hit_ratio(data_generator.valid, emb_model)
        print("HR@10 = {:.4f} , NDCG@10 ={:.4f}".format(hit, ndcg))

        if hit >best_hit:
          best_hit = hit
          torch.save(emb_model.state_dict(), "model/"+ args.model+"_"+ args.domain+".pth")

          count_stop=0
        count_stop+=1
        index+=1

    
    emb_model.load_state_dict(torch.load("model/"+ args.model+"_"+ args.domain+".pth", weights_only=True))
    emb_model.eval()
    test_hit_5, test_ndcg_5, test_hit_10, test_ndcg_10 = test_hit_ratio(data_generator.test, emb_model)
    print("HR@5 = {:.4f} , NDCG@5 ={:.4f}, HR@10 = {:.4f} , NDCG@10 ={:.4f}".format(test_hit_5, test_ndcg_5, test_hit_10, test_ndcg_10))
   
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-6)
    parser.add_argument('--layer_size', type=int, default=3),
    parser.add_argument('--data_path', nargs='?', default="../data/")
    parser.add_argument('--model', nargs='?', default="XSimGCL")
    parser.add_argument('--domain', nargs='?', default="movie")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args.device)

    train(args)





        