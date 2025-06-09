import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from .GraphSCL import Discriminator, AvgReadout, Encoder, Decoder, SCL
from torch_geometric.nn import GAT,GCN
from torch_geometric.utils import dense_to_sparse

class MyModel_gate(nn.Module):
    def __init__(self, in_ge_dim, in_deconv_dim, emb_dim, latent_dim, output_dim, graph_neigh, dropout=0.0, act=F.relu, num_heads = 4, num_layers=1):
        super(MyModel_gate, self).__init__()
        self.graph_neigh = graph_neigh
        
        self.encoder_ge = Encoder(in_ge_dim, emb_dim, dropout, num_layers)
        self.decoder_ge = Decoder(emb_dim, in_ge_dim, dropout, num_layers)
        self.encoder_deconv = Encoder(in_deconv_dim, emb_dim, dropout, num_layers)
        self.decoder_deconv = Decoder(emb_dim, in_deconv_dim, dropout, num_layers)
        
        self.SCL = SCL(emb_dim)
                
        # Gating mechanism
        self.gate = nn.Linear(2*emb_dim, emb_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.decoder_latent = Decoder(emb_dim, in_ge_dim,dropout,num_layers)
        
        self.gat = GAT(emb_dim, latent_dim, num_layers ,output_dim, dropout, heads=num_heads)

    def forward(self, feat_ge, feat_a_ge, feat_deconv, feat_a_deconv, edge_index):
        
        ######### SCL gene expression #########
        # encoding
        emb_ge = self.encoder_ge(feat_ge, edge_index)
        
        emb_a_ge = self.encoder_ge(feat_a_ge, edge_index)
        
        # decoding
        h_ge = self.decoder_ge(emb_ge, edge_index)
        
        # Self Contrastive Learning
        ret_ge, ret_a_ge = self.SCL(emb_ge, emb_a_ge, self.graph_neigh)
        
        ######### SCL deconvolution #########
        # encoding
        emb_deconv = self.encoder_deconv(feat_deconv, edge_index)
        
        emb_a_deconv = self.encoder_deconv(feat_a_deconv, edge_index)
        
        # decoding
        h_deconv = self.decoder_deconv(emb_deconv, edge_index)
        
        # Self Contrastive Learning
        ret_deconv, ret_a_deconv = self.SCL(emb_deconv, emb_a_deconv, self.graph_neigh)
        
        ######### Fusion #########
        # Normalize features
        emb_ge = F.normalize(emb_ge, p=2, dim=-1)
        emb_deconv = F.normalize(emb_deconv, p=2, dim=-1)
        
        emb_concat = torch.cat((emb_ge, emb_deconv), dim=-1)  # Shape: (N, 2*emb_dim)
        
        # gate weights
        gate = self.sigmoid(self.gate(emb_concat))
        
        emb_latent = gate * emb_ge + (1-gate) * emb_deconv
        
        h_latent = self.decoder_latent(emb_latent, edge_index)
        
        ######### Classifier #########
        x = self.gat(emb_latent,edge_index)
            
        return emb_ge, h_ge, ret_ge, ret_a_ge, emb_deconv, h_deconv, ret_deconv,ret_a_deconv, emb_latent, h_latent, x
    
    
