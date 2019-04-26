#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:27:48 2019

@author: jjg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    
    def __init__(self, args):
        super(TextCNN, self).__init__()
        V = args.num_embeddings
        D = args.D
        C = args.C
        Ci = args.Ci
        Co = args.Co
        Ks = args.Ks
        weight_matrix = args.weight
        static = args.static
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(weight_matrix)
        if static:
            self.embed.weight.requires_grad=False
        else:
            self.embed.weight.requires_grad=True
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        #can keep the size by set padding=(kernel_size-1)//2, if stride=1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=((K-1)//2,0)) for K in Ks])
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        x = x.unsqueeze(1)  # (N, Ci, W, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        #ModuleList can act as iterable, or be indexed using ints
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        
        x = torch.cat(x, 1)   #concatenate different feature from different kernel sizes

        x = self.dropout(x)  # (N, len(Ks)*Co)
    
        x = self.fc1(x)  # (N, C)
        return x


class Sim_CNN(nn.Module):
    
    def __init__(self, args):
        super(Sim_CNN, self).__init__()
        V = args.num_embeddings
        D = args.D
        C = args.C
        Ci = args.Ci
        Co = args.Co
        Ks = args.Ks
        weight_matrix = args.weight
        static = args.static
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(weight_matrix)
        if static:
            self.embed.weight.requires_grad=False
        else:
            self.embed.weight.requires_grad=True
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        #can keep the size by set padding=(kernel_size-1)//2, if stride=1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=((K-1)//2,0)) for K in Ks])
        self.conv_sim = nn.Conv2d(Ci, C, (1, D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        #self.conv_sim.weight.requires_grad=False
        self.conv_doc = nn.Conv2d(C, C, (300, 1), stride=(300,1))
        
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc2 = nn.Linear(20, C)
        self.fc3 = nn.Linear(C, C)
        self.fc4 = nn.Linear(len(Ks)*Co+C, C)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        x = x.unsqueeze(1)  # (N, Ci, W, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        
        #random initialization kernel + label vector initialization kernel
        #random kernel
        r1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, W), ...]*len(Ks)

        r1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in r1]  # [(N, Co), ...]*len(Ks)
        
        r1 = torch.cat(r1, 1)   #concatenate different feature from different kernel sizes

        r1 = self.dropout(r1)  # (N, len(Ks)*Co)
        
        r1 = self.fc1(r1)  # (N, C)
        
        #label vector kernel
        r2 = self.conv_sim(x) #(N, 10, W)
        r2 = r2.squeeze(3)
        r2 = F.avg_pool1d(r2, r2.size(2)).squeeze(2) #(N, 10)
        #r2 = F.max_pool1d(r2, r2.size(2)).squeeze(2)     
        return r1+r2
        
    

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.V = args.num_embeddings
        self.D = args.D
        self.C = args.C
        self.layers = args.rnn_layers
        self.drop = args.drop if self.layers > 1 else 0
        weight_matrix = args.weight
        self.static = args.static
        self.bidirectional =  args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        if self.static:
            self.embed.weight.requires_grad=False
        else:
            self.embed.weight.requires_grad=True

        self.rnn = nn.LSTM(               
            input_size=self.D,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )

        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (N, W, D)
        
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
# =============================================================================
#         #you can try other initialization in the following way, but zero init performs better.
#         #2.one init
#         h0 = torch.ones(self.layers*self.num_directions, x.shape[0], self.hidden_size).cuda()
#         c0 = torch.ones(self.layers*self.num_directions, x.shape[0], self.hidden_size).cuda()
#         r_out, (h_n, c_n) = self.rnn(x, (h0,c0))
# =============================================================================
        
        # choose r_out at the last time step
        #print(r_out[:, -1, :].shape) #[batch_size,num_directions*hidden_size]
        out = self.fc(r_out[:, -1, :]) #(batch_size,num_classes)
        return out

class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.V = args.num_embeddings
        self.D = args.D
        self.C = args.C
        self.layers = args.rnn_layers
        self.drop = args.drop if self.layers > 1 else 0
        weight_matrix = args.weight
        self.static = args.static
        self.bidirectional =  args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        if self.static:
            self.embed.weight.requires_grad=False
        else:
            self.embed.weight.requires_grad=True

        self.rnn = nn.GRU(               
            input_size=self.D,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-GRU
        )

        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_0 shape (num_layers*num_directions, batch, hidden_size), here we use zero initialization
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        x = self.embed(x)  # (N, W, D)
        
        #initialization hidden state
        #1.zero init
        r_out, h_n = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose r_out at the last time step
        out = self.fc(r_out[:, -1, :]) #(batch_size,num_classes)
        return out
    