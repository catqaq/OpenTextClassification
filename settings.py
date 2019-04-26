#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:44:54 2019

@author: jjg
"""

#settings
class Settings(object):
    def __init__(self, pre_weights, label_vecs, Dim=300, num_class=10, Cin=1, Cout=256, kernel_size=[3,4,5], dropout=0.5, batch_size=64,num_epochs=100, lr=0.01, weight_decay=0,static=True,batch_normalization=True,hidden_size=64,rnn_layers=1,bidirectional=False):
        
        #general parameters
        self.num_embeddings=len(pre_weights)   #vocab size or num_embeddings
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


#or set parameters by argparse
        
#import os
#import argparse

#parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
#parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
#parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
#parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')

# data 
#parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
#parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
#parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
#parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
#
#parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
#
#parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
#parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
#parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
#parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
#parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
#parser.add_argument('-test', action='store_true', default=False, help='train or test')
#args = parser.parse_args()