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
        
        # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        x = x.unsqueeze(1)  
        
        if self.use_bn:
            x=self.bn2d(x)
        # ModuleList can act as iterable, or be indexed using ints
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, L), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        
        #concatenate different feature from different kernel sizes
        x = torch.cat(x, 1)   

        x = self.dropout(x)  # (N, len(Ks)*Co)
    
        x = self.fc1(x)  # (N, C)
        return x
      


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.V = args.num_embeddings
        self.D = args.D
        self.C = args.C
        self.layers = args.rnn_layers
        self.rnn_drop = args.drop if self.layers > 1 else 0
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
            dropout=self.rnn_drop,            #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
    
        #Orthogonal Initialization, 
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (N, L, D)
    
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose r_out at the last time step or outputs at every time step
        if self.bidirectional:
            #concatenate normal RNN's last time step(-1) output and reverse RNN's last time step(0) output
            #print(r_out[:, -1, :self.hidden_size].size()) #[B, hidden_size]
            out = torch.cat([r_out[:, -1, :self.hidden_size],r_out[:, 0, self.hidden_size:]],1)
        else:
            out = r_out[:, -1, :] #[B, hidden_size*num_directions]
        
        out = self.fc(self.dropout(out))
        
        return out
    

class BiLSTM_LSTM(nn.Module):
    """
    BiLSTM + unidirectional LSTM
    """
    def __init__(self, args):
        super(BiLSTM_LSTM, self).__init__()
        self.V = args.num_embeddings
        self.D = args.D
        self.C = args.C
        self.Ci = args.Ci
        self.layers = args.rnn_layers
        self.drop = args.drop if self.layers > 1 else 0
        weight_matrix = args.weight
        static = args.static
        self.bidirectional =  args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.D,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        
        self.lstm = nn.LSTM(               
            input_size=self.hidden_size*self.num_directions, #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=False               #LSTM
        )
        
        #Orthogonal Initialization
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.hidden_size, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose all time steps' output, i.e. r_out
        
        #lstm, choose last time step 
        r_out, (h_n, c_n) = self.lstm(r_out, None)
        r_out = r_out[:,-1,:]
        r_out = self.fc1(self.dropout(r_out))
            
        return r_out

    
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
        
        # choose r_out at the last time step or outputs at every time step
        if self.bidirectional:
            #concatenate normal RNN's last time step(-1) output and reverse RNN's last time step(0) output
            #print(r_out[:, -1, :self.hidden_size].size()) #[B, hidden_size]
            out = torch.cat([r_out[:, -1, :self.hidden_size],r_out[:, 0, self.hidden_size:]],1)
        else:
            out = r_out[:, -1, :] #[B, hidden_size*num_directions]
        
        out = self.fc(self.dropout(out))
        
        return out

