import ot
import os
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.ensemble import IsolationForest
from scipy.sparse import issparse



def preprocessing(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
    
def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    #adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj

def permutation(features):
    ids = np.arange(features[0].shape[0])
    ids = np.random.permutation(ids)
    
    permuted_features = [feature[ids] for feature in features]
    
    return permuted_features
    
def get_feature_ge(adata,preprocess):# add preprocess argument for simulation data
    if preprocess==True:
        adata_Vars =  adata[:, adata.var['highly_variable_rank'].sort_values().index[:3000]]
    else:
        adata_Vars = adata
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation([feat])[0]
    
    adata.obsm['feat_ge'] = feat
    adata.obsm['feat_a_ge'] = feat_a 
    
    
def get_feature_deconv(adata):
    
    adata_Vars = adata.obsm['deconv']
       
    if isinstance(adata_Vars, pd.DataFrame):
        feat = adata_Vars.values
    else:
        feat = adata_Vars
    
    # data augmentation
    feat_a = permutation([feat])[0]
    
    adata.obsm['feat_deconv'] = feat
    adata.obsm['feat_a_deconv'] = feat_a  
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_SCL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_SCL'] = label_SCL
    

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
def adjacency_to_edge_index(adjacency_matrix):
    # Find indices of non-zero entries in the adjacency matrix
    row, col = torch.nonzero(adjacency_matrix, as_tuple=True)
    # Stack the row and col indices to create the edge_index
    edge_index = torch.stack([row, col], dim=0)
    return edge_index