import os
import hnswlib
import numpy as np
from utils.tool import *
from read_data import *
import anndata as ad

def data_process(k_for_anchor,  
                 coefficient_neighbor, 
                 dataset_name, 
                 coefficient_prune, 
                 k=6, 
                 max_element=95536, 
                 data_path="",
                 scale=True):
    
    train_raw_fp = f'{data_path}/{k_for_anchor}_{coefficient_neighbor}_{coefficient_prune}'
    adata  = ad.read_h5ad(os.path.join(train_raw_fp,"train.h5ad"))
    x_array = adata.to_df().values
    y_array = adata.obs['Group'].values
    # print(f"X shape: {x_array.shape}")
    # print(f"Y shape: {y_array.shape}")
    pair = dict(zip(list(range(len(y_array))),list(y_array)))
    if k > 0:
        neighbors = cal_nn(x_array,y_array, pair,k=k, max_element=max_element)
    else:
        return x_array, y_array, None
    return x_array, y_array, neighbors
    
def cal_nn(x, y, pair, k=500, max_element=95536):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements=max_element, 
                 ef_construction=600, 
                 random_seed=600,
                 M=100)
    
    p.set_num_threads(20)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k = 20)
    neighbors = neighbors[:, 1:]
    distance = distance[:, 1:]

    m, n = y.shape[0], k  
    result_array = np.zeros((m, n), dtype=int)
    for i in range(m):
        count = 0
        for j in range(n*2):
            if count==n:
                continue
            if pair[neighbors[i][j]]==y[i]:
                result_array[i][count] = neighbors[i][j]  
                count+=1
    return result_array

    