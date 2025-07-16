import numpy as np
from tqdm import tqdm

def get_top_N_items(candidate_scores, candidates, top_n):
    candidate_scores = candidate_scores.detach().cpu().numpy()
    top_indices = np.argsort(candidate_scores)[-top_n:][::-1]
    top_candidates = [candidates[i] for i in top_indices]
    return top_candidates

def ranking_evaluation(model, data_a, data_b, phase, is_a, is_b):
    if phase == "test":
        user_data_a = data_a.test
        user_data_b = data_b.test
    elif phase == "valid":
        user_data_a = data_a.valid
        user_data_b = data_b.valid

    hit_a_5, hit_a_10, ndcg_a_5, ndcg_a_10, count_a = 0, 0, 0, 0, 0
    hit_b_5, hit_b_10, ndcg_b_5, ndcg_b_10, count_b = 0, 0, 0, 0, 0

    ua, ia, ub, ib = [], [], [], []

    
    if is_a:
        for i in range(user_data_a.shape[0]):
            row_a = user_data_a.iloc[i]
            candidate_a = row_a["candidates"]
            uid_a = np.full(len(candidate_a), int(row_a["uid"]))
            ua.extend(uid_a)
            ia.extend(candidate_a)

    if is_b:
        for i in range(user_data_b.shape[0]):
            row_b = user_data_b.iloc[i]
            candidate_b = row_b["candidates"]
            uid_b = np.full(len(candidate_b), int(row_b["uid"]))
            ub.extend(uid_b)
            ib.extend(candidate_b)

    scores_a, scores_b = model([ua, ia], [ub, ib], "test")

    start_a, start_b = 0, 0

    if is_a:
        for i in range(user_data_a.shape[0]):
            row_a = user_data_a.iloc[i]
            true_item_a = row_a["iid"]
            candidate_a = row_a["candidates"]

            predictions_a = scores_a[start_a: start_a + len(candidate_a)]
            top_k_rec_a = get_top_N_items(predictions_a, candidate_a, 10)

            index_a = np.where(top_k_rec_a == true_item_a)[0]
        
            if index_a.size > 0:
                index_a = index_a[0]
                if index_a < 5:
                    hit_a_5 += 1
                    ndcg_a_5 += 1 / np.log2(index_a + 2)
                if index_a < 10:
                    hit_a_10 += 1
                    ndcg_a_10 += 1 / np.log2(index_a + 2)

            count_a += 1
            start_a += len(candidate_a)

    if is_b:
        for i in range(user_data_b.shape[0]):
            row_b = user_data_b.iloc[i]
            true_item_b = row_b["iid"]
            candidate_b = row_b["candidates"]

            predictions_b = scores_b[start_b: start_b + len(candidate_b)]
            top_k_rec_b = get_top_N_items(predictions_b, candidate_b, 10)

            index_b = np.where(top_k_rec_b == true_item_b)[0]
        
            if index_b.size > 0:
                index_b = index_b[0]
                if index_b < 5:
                    hit_b_5 += 1
                    ndcg_b_5 += 1 / np.log2(index_b + 2)
                if index_b < 10:
                    hit_b_10 += 1
                    ndcg_b_10 += 1 / np.log2(index_b + 2)

            count_b += 1
            start_b += len(candidate_b)

    if is_a and is_b:
        results_a = [hit_a_5 / count_a, ndcg_a_5 / count_a, hit_a_10 / count_a, ndcg_a_10 / count_a]
        results_b = [hit_b_5 / count_b, ndcg_b_5 / count_b, hit_b_10 / count_b, ndcg_b_10 / count_b]
        return results_a, results_b
    else:
        if is_a:
            results_a = [hit_a_5 / count_a, ndcg_a_5 / count_a, hit_a_10 / count_a, ndcg_a_10 / count_a]
            return results_a
        if is_b:
            results_b = [hit_b_5 / count_b, ndcg_b_5 / count_b, hit_b_10 / count_b, ndcg_b_10 / count_b]
            return results_b
