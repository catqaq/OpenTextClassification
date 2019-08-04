#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:27:48 2019

@author: jjg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        if static in [1, 2]:
            self.embed.weight.requires_grad=False
        else:
            self.embed.weight.requires_grad=True
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        #can keep the size by set padding=(kernel_size-1)//2, if stride=1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=((K-1)//2,0)) for K in Ks])
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        #ModuleList can act as iterable, or be indexed using ints
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, L), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        
        x = torch.cat(x, 1)   #concatenate different feature from different kernel sizes

        x = self.dropout(x)  # (N, len(Ks)*Co)
    
        x = self.fc1(x)  # (N, C)
        return x


class Hybrid_CNN(nn.Module):
    
    def __init__(self, args):
        super(Hybrid_CNN, self).__init__()
        V = args.num_embeddings
        self.L = args.L
        D = args.D
        self.C = args.C
        Ci = args.Ci
        Co = args.Co
        Ks = args.Ks
        weight_matrix = args.weight
        static = args.static
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(weight_matrix)
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        #can keep the size by set padding=(kernel_size-1)//2, if stride=1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=((K-1)//2,0)) for K in Ks])
        self.conv_sim = nn.Conv2d(Ci, self.C, (1, D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False
        self.conv_doc = nn.Conv2d(self.C, self.C, (300, 1), stride=(300,1))
        
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(len(Ks)*Co, self.C)
        self.fc2 = nn.Linear(20, self.C)
        self.fc3 = nn.Linear(self.L, 1)
        self.fc4 = nn.Linear(len(Ks)*Co+self.C, self.C)
        self.fc5 = nn.Linear(len(Ks)*Co+self.L*self.C, self.C)
        self.fc6 = nn.Linear(self.L*self.C, self.C)
        self.fc7 = nn.Linear(self.C, self.C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
            
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        
        r1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, L), ...]*len(Ks)
        r1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in r1]  # [(N, Co), ...]*len(Ks) 
        r1 = torch.cat(r1, 1)   #(N, len(Ks)*Cout)
        r1 = self.dropout(r1)  
        r1 = self.fc1(r1)  # (N, C)
        
        x = self.conv_sim(x).squeeze() #(N, 10, L)
        x = [self.fc3(self.dropout(i)) for i in torch.chunk(x, self.C, 1)]
        x = torch.cat(x, 1).squeeze()
        
        return x+r1
        

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
        Ci = args.Ci
        weight_matrix = args.weight
        static = args.static
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        self.pe1 = PositionalEncoding(self.D, args.drop)
        self.pe2 = PE(self.L, max_len=self.V)
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        self.conv_sim = nn.Conv2d(Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.construct = nn.Conv2d(Ci, self.C, (self.L, 1)) #learn doc constructure
        self.multi_cons = nn.Conv2d(Ci, self.C*10, (self.L, 1))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False
        
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.L*self.C, self.C)
        self.fc2 = nn.ModuleList([nn.Linear(self.L, 1) for i in range(self.C)])
        self.fc3 = nn.Linear(self.L, 1)
        self.fc4 = nn.Linear(self.L, self.C)
        self.fc5 = nn.Linear(self.C*10, self.C)

    def forward(self, x):
        pe = self.pe2(x)   # (N, L)
        pe = pe.unsqueeze(1).unsqueeze(3)
        pe = self.multi_cons(pe).squeeze()
        x = self.embed(x)  # (N, L, D)
        
        #x = self.pe1(x)     #word embedding + position embedding
            
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        
        x = self.conv_sim(x).squeeze() #(N, C, L)
        x = [self.fc3(self.dropout(i)) for i in torch.chunk(x, self.C, 1)]
        x = torch.cat(x, 1).squeeze()
        x = x + self.fc5(self.dropout(pe))
        return x

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

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) #max_len: max doc length
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *-(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1)]).clone().detach()#为啥+x? 这里的x是词嵌入后的结果，因此这里其实是词嵌入+位置嵌入
        return self.dropout(x)


class PE(nn.Module):
    "a simple PE version without considering dimention"
    def __init__(self, L, max_len=5000):
        super(PE, self).__init__()
        self.L = L
        self.max_len = max_len
        pass #any better way?
        
    
    def forward(self, x):
        x = torch.sin(x.float()/self.max_len) #div_term:1, L^0.5, L, len(vocab)?
        return x
        