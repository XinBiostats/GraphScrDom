import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, GATConv, GAT

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1)
    
class Encoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_layers=1):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Define the list of GCN layers
        self.gcn_layers = torch.nn.ModuleList()
        
        # Add the first GCN layer (input to hidden)
        self.gcn_layers.append(GCNConv(in_dim, out_dim))
        
        # Add additional GCN layers (hidden to hidden) if num_layers > 1
        for _ in range(1, num_layers):
            self.gcn_layers.append(GCNConv(out_dim, out_dim))
        
    def forward(self, feat, edge_index):
        z = feat
        
        # Pass through each GCN layer
        for layer in self.gcn_layers:
            z = layer(z, edge_index)
            z = F.relu(z)  # Apply activation
            z = F.dropout(z, self.dropout, training=self.training)  # Apply dropout

        return z
    
class Decoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0, num_layers=1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Define the list of GCN layers for the decoder
        self.gcn_layers = torch.nn.ModuleList()
        
        # First layer (latent to hidden)
        self.gcn_layers.append(GCNConv(in_dim, out_dim))
        
        # Intermediate layers (hidden to hidden)
        for _ in range(1, num_layers - 1):
            self.gcn_layers.append(GCNConv(out_dim, out_dim))
        
        # Last layer (hidden to output)
        self.gcn_layers.append(GCNConv(out_dim, out_dim))

    def forward(self, x, edge_index):
        # Pass through each GCN layer
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:  # Apply activation and dropout for intermediate layers only
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x    

    
class SCL(nn.Module):
    def __init__(self, out_features):
        super(SCL, self).__init__()

        self.disc = Discriminator(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def forward(self, emb, emb_a, graph_neigh):
        g = self.read(emb, graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, graph_neigh)
        g_a = self.sigm(g_a)  
        
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        return ret, ret_a