import os
import h5py
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])

        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], 
                                            exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), 
                                            shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))

    return mat, obs, var, uns


def prepro(filename):
    data_path = os.path.join(filename, "data.h5")
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)

    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())

    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)

    return X, cell_label, cell_type, var

def prepare_npz(file_name):
    input_path = os.path.join(file_name,"filtered_Counts.npz")
    label_path = os.path.join(file_name,"annoData.txt")
    gene_path = os.path.join(file_name,"genes.txt")
    X= sp.sparse.load_npz(input_path)
    X = X.toarray()
    label = pd.read_csv(label_path,header=0,index_col=None,delimiter='\t')
    genes = pd.read_csv(gene_path,header=None)
    # if "cellIden3" in label.columns.to_list():
    #     Y = label['cellIden3'].to_numpy()
    # # elif "cellIden0" in label.columns.to_list(): 
    # #     Y = label['cellIden0'].to_numpy()
    # else:
    #     Y = label['cellIden'].to_numpy()
    if 'cellAnno' in label.columns.to_list():
        cell_name = label['cellAnno'].to_numpy()
    elif 'Main_cluster_name' in label.columns.to_list():
        cell_name = label['Main_cluster_name'].to_numpy()
    else:
        cell_name = label['celltype'].to_numpy()
    cell_type, Y = np.unique(cell_name, return_inverse=True)    
    return X, Y, cell_type,list(genes.iloc[:,0])

def prepare_nested_h5(file_name):
    X, Y, cell_type, var = prepro(file_name)

    X = np.ceil(X).astype(np.int32)

    return X, Y, cell_type, list(var)

def prepare_h5ad(file_name):
    adata = sc.read_h5ad(os.path.join(file_name, "data.h5ad"))
    X = sp.sparse.csr_matrix(adata.X)
    X = X.toarray()

    
    if "cell_type" not in adata.obs.columns.to_list():
        cell_name = np.array(adata.obs["clusters"])
    else:
        cell_name = np.array(adata.obs["cell_type"])
    cell_type, Y = np.unique(cell_name, return_inverse=True)

    return X, Y, cell_type

def prepare_h5(file_name):
    import h5py
    data_mat = h5py.File(os.path.join(file_name, "data.h5"), "r")

    X = np.array(data_mat['X'])
    Y = np.array(data_mat['Y'])

    cell_type, Y = np.unique(Y, return_inverse=True)

    return X, Y, cell_type

def prepare_h5_new(file_name):
    import anndata as ad 
    adata = ad.read_h5ad(os.path.join(file_name,"data.h5"))
    X = adata.X
    cell_type, Y = np.unique(np.array(adata.obs['label']),return_inverse=True)
    return X, Y, cell_type

