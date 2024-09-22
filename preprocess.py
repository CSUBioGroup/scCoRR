import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from imblearn.over_sampling import SMOTE
from collections import Counter
from utils.tool import *
from utils.util import yaml_config_hook, normalize,  find_anchor_cell, adata_to_save, save_adata,type_to_num
import os
import argparse
import warnings

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_baron_human.yaml")
    # config = yaml_config_hook("config/config_Adam.yaml")


    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    data_path = os.path.join(args.data_path,f'{args.k_for_anchor}_{args.coefficient_neighbor}_{args.coefficient_prune}')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    file_path = os.path.join(args.root_dir,args.data_type,args.dataset_name)
    if args.data_type == "npz":
        X, Y, cell_type, _ = prepare_npz(file_path)
    elif args.data_type == "h5_nested":
        X, Y, cell_type, _ = prepare_nested_h5(file_path)
    elif args.data_type == "h5":
        X, Y, cell_type = prepare_h5(file_path)
    elif args.data_type == "h5ad":
        X, Y, cell_type = prepare_h5ad(file_path)
    elif args.data_type == "h5_new":
        X, Y, cell_type = prepare_h5_new(file_path)
    else:
        raise Exception("Please Input Proper Data Type")

    adata = ad.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['Group'] = adata.obs['Group'].astype('category')
    adata = normalize(adata,args.num_genes)
    adata= adata[:,adata.var.highly_variable]
    sc.tl.rank_genes_groups(adata, groupby='Group', method='wilcoxon')
    print(f"Gene num is: {adata.shape[1]}")
    pair_cell = dict(zip(list(range(len(adata.obs['Group']))),list(adata.obs['Group'])))
    if isinstance(cell_type[0], bytes):
        cell_type = [ct.decode('utf-8')  for ct in cell_type]
    pair_celltype = dict(zip(range(len(cell_type)),cell_type))
    sc.tl.pca(adata, 
        # n_comps=50,
        svd_solver='arpack',
        random_state=0)
    sc.pp.neighbors(adata, 
                    n_neighbors=20, 
                    n_pcs=40,
                    use_rep='X'
                    )
    adata.obs['cell_type'] = adata.obs['Group']
    adata.obs['cell_type'] = adata.obs['cell_type'].replace(pair_celltype)
    umap_embedding = type_to_num(adata)
    
    #anchor
    marker_df = pd.DataFrame(adata.uns ['rank_genes_groups']['names'])
    marker_df = marker_df[:args.top_markers]
    centers = {}
    for ct in range(args.classnum):
        cell_indices = adata.obs_names[adata.obs['Group']== ct].astype(int)
        centers[ct] = np.mean(umap_embedding[cell_indices], axis=0)
    result_df = find_anchor_cell(pair_cell,marker_df,adata,
                                args.k_for_anchor,args.coefficient_neighbor,args.coefficient_prune,args.classnum,centers,umap_embedding)
    list_t = [key for key, value in Counter(result_df['label']).items() if value > 10]
    result_df  = result_df[result_df['label'].isin(list_t)]
    print("Anchor completed")

    train_raw_path = os.path.join(data_path,"train_raw.h5ad")
    all_path = os.path.join(data_path,"all.h5ad")
    adata_train_raw = adata[result_df.index,:]
    original_classes = [key for key, value in Counter(adata_train_raw.obs['Group']).items()]
    remaining_classes = np.arange(len(adata_train_raw.obs['Group'].unique()))
    class_mapping = {original_class: new_class for original_class, new_class in zip(original_classes,remaining_classes)}
    new_labels_train_raw = list([class_mapping[label] for label in adata_train_raw.obs['Group']])
    adata_train_raw = adata_to_save(adata_train_raw)
    adata_train_raw.obs['Group'] = new_labels_train_raw
    adata_train_raw.obs['Group'] = adata_train_raw.obs['Group'].astype('category')
    adata_train_raw.obsm['X_umap'] = umap_embedding[list(adata_train_raw.obs.index.astype(int))]
    adata.obsm['X_umap'] = umap_embedding
    adata_all =adata[adata.obs['Group'].isin(original_classes)]
    adata_all = adata_to_save(adata_all)
    new_labels_all = list([class_mapping[label] for label in adata_all.obs['Group']])
    adata_all.obs['Group'] = new_labels_all
    adata_all.obs['Group'] = adata_all.obs['Group'].astype('category')
    adata_all.obsm['X_umap'] = umap_embedding[list(adata_all.obs.index.astype(int))]
    save_adata(train_raw_path,adata_to_save(adata_train_raw))
    mask = [cell_name not in result_df.index for cell_name in range(adata.shape[0])]
    adata_test = adata[np.array(mask),:]
    adata_test = adata_test[adata_test.obs['Group'].isin(list(original_classes))]
    adata_test = adata_to_save(adata_test)
    new_labels_test = list([class_mapping[label] for label in adata_test.obs['Group']])
    adata_test.obs['Group'] = new_labels_test
    adata_test.obs['Group'] = adata_test.obs['Group'].astype('category')
    adata_test.obsm['X_umap'] = umap_embedding[list(adata_test.obs.index.astype(int))]
    save_adata(all_path,adata_all) 
    X = adata_train_raw.to_df()
    y = list(adata_train_raw.obs['Group'].astype(int))
    dict_filter = adata_train_raw.obs['Group'].value_counts().to_dict()
    
    #smote
    count_mean = int(result_df.shape[0]/adata_train_raw.obs['Group'].nunique())
    sampling_strategy = {key: count_mean for key, value in dict_filter.items() if value < count_mean}
    k_neighbors = min(Counter(adata_train_raw.obs['Group']).values())-1
    if k_neighbors==0:
        k_neighbors += 1
    smo = SMOTE(sampling_strategy=sampling_strategy,k_neighbors=k_neighbors, random_state=42)
    X_smo, y_smo = smo.fit_resample(X, y)
    print(f"Train:Test -> {adata_train_raw.shape[0]}:{adata_test.shape[0]}   {adata_train_raw.shape[0]/adata_test.shape[0]}")
    
    #save
    adata_train_sample = ad.AnnData(X_smo,dtype=np.float32)
    adata_train_sample.obs['Group'] = y_smo
    adata_train_sample.obs['Group'] = adata_train_sample.obs['Group'].astype('category')
    adata_test_new = adata_to_save(adata_test)
    train_path = os.path.join(data_path,"train.h5ad")
    test_path = os.path.join(data_path,"test.h5ad")
    save_adata(test_path,adata_test_new)
    save_adata(train_path,adata_train_sample)
    print("Prepare train and test completed")
    
if __name__ == "__main__":
    main()