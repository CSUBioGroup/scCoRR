import os
import yaml
import random
import numpy as np
import scanpy as sc
import torch
import hnswlib
import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import anndata as ad
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def save_model(model_path, model, optimizer, current_epoch):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
    
def normalize(adata, num_genes):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata=adata,n_top_genes=num_genes)
    return adata

def find_neighbors_hnswlib(data,k):
    ids = np.arange(data.shape[0])
    max_element = 95536
    p = hnswlib.Index(space = 'cosine', dim = data.shape[1]) 
    p.init_index(max_elements = max_element, ef_construction = 600, M = 100)
    p.set_num_threads(20)
    p.set_ef(600)
    p.add_items(data, ids)
    neighbors, distances = p.knn_query(data, k = k)
    return neighbors,distances

def filter_center(index,label,centers,umap_embedding):
    cell_coor = umap_embedding[index]
    distances = {}
    for other_cell_type, other_center in centers.items():
        distance = pairwise_distances([cell_coor], [other_center])[0, 0]
        distances[other_cell_type] = distance
    if any(distance < distances[label] for distance in distances.values()):
        return False
    else:
        return True
    
def find_anchor_cell(paired,marker_df,adata, 
                    k_for_anchor,coefficient_neighbor,coefficient_prune,classnum,
                    centers,umap_embedding):
    df_a = adata.to_df()
    result_df = pd.DataFrame()
    for i in tqdm(range(classnum)):
#1
        neighbors,distances = find_neighbors_hnswlib(df_a.loc[:,list(marker_df.iloc[:,i])],k_for_anchor)
        neighbor_df = pd.DataFrame(neighbors)
        neighbor_df_label = neighbor_df.applymap(lambda x: paired.get(x, x))
        first_element = neighbor_df_label.iloc[:, 0].values  
        equal_count_per_row = np.sum(neighbor_df_label.values[:, 1:] == first_element[:, None], axis=1)
        neighbor_df_label['equal'] = equal_count_per_row
        neighbor_df_label['label'] = list(adata.obs['Group'].astype(int))
        indexes = neighbor_df_label[neighbor_df_label['equal'] > k_for_anchor*coefficient_neighbor].index
        neighbor_df_label = neighbor_df_label.loc[neighbor_df_label.index.isin(indexes)]
        neighbor_df_label = neighbor_df_label[neighbor_df_label['label']==i]
#2
        distances_df = pd.DataFrame(distances)
        distances_df = distances_df.loc[distances_df.index.isin(neighbor_df_label.index)]
        mean_values = distances_df.iloc[:, 1:].mean(axis=1)
        distances_df[0] = mean_values
        distances_df[1] = list(adata[neighbor_df_label.index].obs['Group'])
        distances_df = distances_df.loc[:,[0,1]]
        distances_df.columns = ['mean','label']
        temp_df=distances_df.sort_values(by='mean').head(int(distances_df.shape[0]*coefficient_prune))
#3
        mask = temp_df.index.to_series().apply(filter_center,args=[i,centers,umap_embedding,])
        temp_df.drop(temp_df[~mask].index,inplace=True)
        result_df = pd.concat([result_df,temp_df])
    return result_df

def adata_to_save(adata):
    adata_new = ad.AnnData(adata.to_df().to_numpy())
    adata_new.obs.index = adata.obs.index
    adata_new.var.index = adata.var.index
    adata_new.obs['Group'] = list(adata.obs['Group'])
    adata_new.obs['Group'] = adata_new.obs['Group'].astype('category')
    return adata_new

def save_adata(path,adata):
    adata.write(path,compression="gzip")

def type_to_num(adata):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.36, hspace=0.3)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    colors = ['cell_type','Group']
    titles = ['Cell_type','Number'] 
    sc.tl.umap(adata)
    for i in range(2):
        sc.pl.umap(adata, color=colors[i], title=titles[i] ,ax=axs[i], show=False, palette='tab20')
    return adata.obsm['X_umap']
    