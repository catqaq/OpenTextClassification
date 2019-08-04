#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:57:17 2019

@author: jjg
"""
import torch
import torch.nn as nn

class Sim_CNN(nn.Module):
    """
    use Sim_CNN to learn label presentation and compute similarities to help classification
    """
    
    def __init__(self, args):
        super(Sim_CNN, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
        self.D = args.D
        self.C = args.C
        self.Ks = args.Ks #To compare with TextCNN, we use the same kernel size
        Ci = args.Ci
        weight_matrix = args.weight
        static = args.static
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        #C label embedding kernels with size(1,D)
        self.conv_sim = nn.Conv2d(Ci, self.C, (1, self.D)) 
        #initialize with label_vecs obtained by sum embeddings
        self.conv_sim.weight = nn.Parameter(args.label_vecs) 
        #static=1,2,3,4, where static=4 refers to update word embedding and label embedding 
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False
        
        self.dropout = nn.Dropout(args.drop)
        self.fcs = nn.ModuleList([nn.Linear(self.L-k, 1) for k in self.Ks])
        self.fc = nn.Linear(self.C*len(self.Ks), self.C)

    def forward(self, x):
        x = self.embed(x)  
            
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        
        #combination
        x=[torch.cat([torch.sum(torch.index_select(x,dim=2,index=torch.tensor(range(i,i+k)).cuda()),dim=2,keepdim=True) for i in range(self.L-k)],dim=2) for k in self.Ks]
        #print(x[0].size()) #[256, 1, 97, 300]
        
        #compute the similarity
        x = [self.conv_sim(i).squeeze() for i in x]
        #print(x[0].size()) #[256, 14, 97]
        
        x = [torch.cat([self.fcs[i](self.dropout(j)) for j in torch.chunk(x[i], self.C, 1)], 1).squeeze() for i in range(len(x))]
        #print(x[0].size()) #[N, C]
        
        x = torch.cat(x, 1).squeeze() #[N, len(Ks)*C]
        x = self.fc(x)      #[N, C]
        return x