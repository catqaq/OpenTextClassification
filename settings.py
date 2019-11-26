#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:44:54 2019

@author: jjg
"""

#settings
class Settings(object):
    def __init__(self, pre_weights, label_vecs, L, Dim=300, num_class=10, Cin=1, Cout=256, kernel_size=[3,4,5], dropout=0.5, batch_size=64,num_epochs=100, lr=0.01, weight_decay=0,static=True,batch_normalization=True,hidden_size=64,rnn_layers=1,bidirectional=False):
        
        #general parameters
        self.num_embeddings=len(pre_weights)   #vocab size or num_embeddings
        self.L=L
        self.D=Dim                       #embedding dimention
        self.C=num_class                 #how many classes
        self.drop=dropout                #probability of an element to be zeroed
        self.epochs=num_epochs           #training epochs
        self.lr=lr                       #learning rate
        self.weight=pre_weights          #pretrained word vector matrix
        self.static=static               #static for not update embeddings, not static for update embeddings
        self.batch_size=batch_size
        self.l2=weight_decay             #l2 regularization
        self.use_bn=batch_normalization  #use batch normalization or not
        
        #cnn parameters
        self.label_vecs=label_vecs       #use label vectors to initialize the sim_kernel
        self.Ci=Cin                      #in_channels
        self.Co=Cout                     #kernel numbers (out_channels)
        self.Ks=kernel_size              #kernel height, a list of varying kernel height
                
        #rnn parameters
        self.hidden_size=hidden_size     #rnn hidden size
        self.rnn_layers=rnn_layers       #stack how many rnn layers
        self.bidirectional=bidirectional #bi-rnn