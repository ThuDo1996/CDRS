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
- **DirectAU** : is a graph-based method that utilize alignment and uniformity to optimize model
- **XSimGCL** : The combination of graph neural network as base model with contrastive learning task to enhance graph representation learning.

**Cross-domain methods**
- **Bi-TGCF**: is a graph-based method that performs the knowledge transfer at each graph encoder layer.

You can switch between models in `main.py` by changing the model selection parameter.


#### Experimental Results 
> Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are used as evaluation metrics
|          |          Movies     |          CDs        |
|  Method  |----------|----------|----------|----------|
|          |    HR@10 |  NDCG@10 |   HR@10  |  NDCG@10 |
|----------|----------|----------|----------|----------|
| DirectAU |  0.2022  |  0.1193  |  0.1791  |  0.0968  |
| XSimGCL  |  0.2791  |  0.1617  |  0.1950  |  0.1084  |
|   CCDR   |  0.2271  |  0.1344  |  0.2325  |  0.1344
|  Bi-TGCF |  0.4582  |  0.2629  |  0.4572  |  0.2633  |
- **LightGCN** aggregates information uniformly from neighboring nodes, which inadvertently amplifies the influence of popular nodes due to their high connectivity. This leads to ***popularity bias***, skewing the representation learning process. As a result, popular items dominate the recommendation lists, reducing the visibility of less popular items and causing performance imbalance that negatively impacts overall effectiveness.
- **SGL** mitigate this issue by incorporating ***contrastive learning***, which enhances representation learning by pulling similar nodes closer while pushing dissimilar ones apart, thus improving generalization. 

#### References:

[1] He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 639-648).

[2] Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021, July). Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval (pp. 726-735).
