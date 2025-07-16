
## Cross-domain Recommendation Systems
Cold-start remains a fundamental challenge in recommendation systems, especially when new users or items have little to no interaction history. Cross-domain recommendation addresses this issue by transferring knowledge from one domain to another. For instance, if Anna shows a preference for romantic movies, this information can be leveraged in the CDs domain to recommend romantic songs that align with her tastes.

### 🔧 Model Training
We implement several recommendation methods which can be classified into two main categories: **single-domain** and **cross-domain**. We conduct experiments on the real-world Amazon datasets Movies and CDs. Users with fewer than 10 interactions are removed to ensure data quality. To simulate cold-start scenarios, we randomly select 20% of overlapping users as cold-start users in domain Movies, and a separate 20% as cold-start users in domain CDs. Cold-start users in Movies are treated as training users in CDs, and vice versa. For domain-specific users, 10% are randomly sampled to serve as cold-start users.

|  Domain  |  #users  |  #items  |#interactions| #overlap | %overlap |
|----------|----------|----------|-------------|----------|----------|
|  Movies  |  58337   |  58024   |  1123403    |   8742   |  15%     |
|    CDs   |  20681   |  70655   |   457699    |   8742   |  42%     |

#### 📁 Project Structure

- `single-domain`: Contains single-domain recommendation model such as XSimGCL, DirectAU

- `cross-domain`: Contains cross-domain recommendation model such as Bi-TGCF, CCDR  


#### 🧠 Model Descriptions

**Single-domain methods**:
- **DirectAU** [1]: is a graph-based method that utilize alignment and uniformity to optimize model
- **XSimGCL** [2]: The combination of graph neural network as base model with contrastive learning task to enhance graph representation learning.

**Cross-domain methods**
- **Bi-TGCF** [3]: is a graph-based method that performs the knowledge transfer at each graph encoder layer.
- **CCDR** [4]: is the graph-based method that reduce the distribution gap between domains by optimizing the contrastive learning loss based on overlapping users.

You can switch between models in `main.py` by changing the model selection parameter.


#### Experimental Results 
> Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are used as evaluation metrics

| Method    | Movies HR@10 | Movies NDCG@10 | CDs HR@10 | CDs NDCG@10 |
|-----------|--------------|----------------|-----------|-------------|
| DirectAU  | 0.2022       | 0.1193         | 0.1791    | 0.0968      |
| XSimGCL   | 0.2191       | 0.1617         | 0.1950    | 0.1084      |
| Bi-TGCF   | 0.4582       | 0.2629         | 0.4572    | 0.2633      |
| CCDR      | 0.2271       | 0.1344         | 0.2325    | 0.1344      |

**Cross-domain** recommendation methods, such as CCDR and Bi-TGCF, outperform **single-domain** methods like DirectAU and XSimGCL, demonstrating the effectiveness of knowledge transfer in addressing the cold-start problem, with improvements of up to 120%. Bi-TGCF achieves superior performance by performing deep knowledge transfer through information sharing at each graph encoder layer, whereas CCDR transfers knowledge only at the final layer.

#### References:

[1] C. Wang, Y.Yu, W.Ma, M.Zhang, C.Chen, Y.Liu, and S.Ma, Towards representation alignment and uniformity in collaborative filtering, In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pp. 1816-1825, 2022.

[2] J.Yu, X.Xia, T.Chen, L.Cui, N.Q.V.Hung, and H.Yin, XSimGCL: Towards extremely simple graph contrastive learning for recommendation, IEEE Transactions on Knowledge and Data Engineering, 36(2), 913-926, 2023.

[3]M. Liu, J. Li, G. Li, and  P. Pan,  Cross domain recommendation via bi-directional transfer graph collaborative filtering networks, In Proceedings of the 29th ACM International Conference on Information and Knowledge management, pp. 885-894, 2020.

[4] R. Xie, Q. Liu, L. Wang, S. Liu, B. Zhang, and L. Lin, Contrastive cross-domain recommendation in matching, in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4226-4236,  2022.
