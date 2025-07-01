import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from .preprocess import *
from .model import MyModel_gate
from .loss import SCLLoss,UnsupLoss,SupLoss
from .utils import remap_labels,sigmoid_rampup

class GraphScrDom():
    def __init__(self,
                 adata,
                 preprocess = True,
                 seed = 41,
                 device = torch.device('cpu'),
                 learning_rate = 0.001,
                 weight_decay = 0,
                 dropout = 0,
                 epochs = 600,
                 num_neighbor = 3, 
                 emb_dim = 64,
                 latent_dim = 64,
                 num_heads = 4,
                 num_layers = 1,
                 verbose = False
                ):
        
        self.adata = adata.copy()
        self.preprocess = preprocess
        #self.dataset = self.adata.uns['dataset']
        #self.sample = self.adata.uns['sample']
        self.seed = seed
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.num_neighbor = num_neighbor
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.verbose = verbose
        
        fix_seed(seed = self.seed)

        self.adata = self.adata[self.adata.obs['annotation'].notna()].copy()
        # 1. Data Preparation

        if 'highly_variable' not in self.adata.var.keys() and self.preprocess==True: # add preprocess argument for simulation data
            preprocessing(self.adata)    

        if 'adj' not in self.adata.obsm.keys(): 
            construct_interaction(self.adata, self.num_neighbor)

        if 'label_SCL' not in self.adata.obsm.keys():    
            add_contrastive_label(self.adata)

        if 'feat_ge' not in self.adata.obsm.keys():
            get_feature_ge(self.adata, self.preprocess)

        if 'feat_deconv' not in self.adata.obsm.keys():
            get_feature_deconv(self.adata)
            
        self.in_ge_dim = self.adata.obsm['feat_ge'].shape[1]
        self.in_deconv_dim = self.adata.obsm['feat_deconv'].shape[1]
            
        self.features_ge = torch.FloatTensor(self.adata.obsm['feat_ge'].copy()).to(self.device)
        self.features_a_ge = torch.FloatTensor(self.adata.obsm['feat_a_ge'].copy()).to(self.device)
        self.features_deconv = torch.FloatTensor(self.adata.obsm['feat_deconv'].copy()).to(self.device)
        self.features_a_deconv = torch.FloatTensor(self.adata.obsm['feat_a_deconv'].copy()).to(self.device)
        self.label_SCL = torch.FloatTensor(self.adata.obsm['label_SCL']).to(self.device)
        self.adj = torch.FloatTensor(preprocess_adj(self.adata.obsm['adj']))
        self.edge_index = adjacency_to_edge_index(self.adj).to(self.device)
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        
        self.inds_sup = torch.FloatTensor(np.where(~self.adata.obs['scribble'].isna())[0]).long().to(self.device) #supervise
        self.inds_unsup = torch.FloatTensor(np.where((self.adata.obs['scribble'].isna()))[0]).long().to(self.device)
        
        label,_ = remap_labels(torch.FloatTensor(self.adata.obs['annotation']))
        self.label = label.long().to(self.device)
        
        self.loss1=[]
        self.loss2=[]
        self.loss3=[]
        self.loss4=[]
        self.loss5=[]
        
        self.output_dim = self.adata.obs['annotation'].dropna().nunique()
        print('processing done')
        
        # 2. Train
    def train(self, save_dir, patience=10):
        os.makedirs(save_dir, exist_ok=True)
        

        self.model = MyModel_gate(self.in_ge_dim, self.in_deconv_dim, self.emb_dim, self.latent_dim, self.output_dim, self.graph_neigh, dropout = self.dropout, act = F.relu, num_heads = self.num_heads, num_layers = self.num_layers).to(self.device)

        loss_SCL = SCLLoss(weight_mse=1,weight_scl=0.1)
        loss_Unsup = UnsupLoss()
        loss_Sup = SupLoss()
        loss_MSE = nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        
        
        min_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(save_dir,'best_model.pth')
        for epoch in (range(self.epochs)):
            self.model.train()
            optimizer.zero_grad()

            emb_ge, h_ge, ret_ge, ret_a_ge, emb_deconv, h_deconv, ret_deconv,ret_a_deconv, emb_latent, h_latent, x = self.model(self.features_ge, self.features_a_ge, self.features_deconv, self.features_a_deconv, self.edge_index)
            

            # SCL GE
            loss1 = loss_SCL(self.features_ge, h_ge, ret_ge, ret_a_ge, self.label_SCL)
            
            
            # SCL Deconv
            loss2 = loss_SCL(self.features_deconv, h_deconv, ret_deconv, ret_a_deconv, self.label_SCL)

            # Classification Loss
            # unsupervise loss
            loss3 = loss_Unsup(x, self.inds_unsup)
            # supervise loss
            loss4 = loss_Sup(x, self.inds_sup, self.label)
            
            loss5 = loss_MSE(self.features_ge,h_latent)
             
            loss = loss1 + loss2 + sigmoid_rampup(epoch,self.epochs*0.5)*loss3 + loss4 + loss5
            cls_loss = loss3 + loss4
            
            self.loss1.append(loss1.item())
            self.loss2.append(loss2.item())
            self.loss3.append(loss3.item())
            self.loss4.append(loss4.item())
            self.loss5.append(loss5.item())
            
            if cls_loss.item() < min_loss:
                min_loss = cls_loss.item()
                torch.save(self.model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
    
            if self.verbose and epoch % 20 == 0:
                print(f"Epoch: {epoch}",
                      f"SCL GE Loss: {loss1.item():.6f}",
                      f"SCL Deconv Loss: {loss2.item():.6f}",
                      f"Unsup Loss: {loss3.item():.6f}",
                      f"Sup Loss: {loss4.item():.6f}",
                      f"Latent Loss: {loss5.item():.6f}"
                     )

            loss.backward()
            optimizer.step()
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best classification loss: {min_loss:.6f}")
                break
    
        self.model.load_state_dict(torch.load(best_model_path))
        with torch.no_grad():
            self.model.eval()
            emb_ge, h_ge, ret_ge, ret_a_ge, emb_deconv, h_deconv, ret_deconv,ret_a_deconv, emb_latent,h_latent, x = self.model(self.features_ge, self.features_a_ge, self.features_deconv, self.features_a_deconv, self.edge_index)
            _, prediction = torch.max(x,1)
            prediction = prediction.cpu().numpy()
            
            self.adata.obs['prediction'] = pd.Categorical(prediction)
            
            #output final embedding
            self.adata.obsm['final_emb'] = emb_latent.cpu().numpy()
            
            #output ge/deconv embedding
            self.adata.obsm['ge_emb'] = emb_ge.cpu().numpy()
            self.adata.obsm['deconv_emb'] = emb_deconv.cpu().numpy()
            
            #output ge/deconv reconstruction
            self.adata.obsm['ge_recon'] = h_ge.cpu().numpy()
            self.adata.obsm['deconv_recon'] = h_deconv.cpu().numpy()
            
            del self.adata.obsm['graph_neigh']
            
        return self.adata
            
