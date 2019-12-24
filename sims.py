import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimCNN(nn.Module):
    """
    use SimCNN to learn label presentation and compute similarities to help classification
    """
    
    def __init__(self, args):
        super(SimCNN, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
        self.D = args.D
        self.C = args.C
        Ci = args.Ci
        weight_matrix = args.weight
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        self.conv_sim = nn.Conv2d(Ci, self.C, (1, self.D)) #label_vec_kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if args.static:
            self.embed.weight.requires_grad=False
        if args.sim_static:
            self.conv_sim.weight.requires_grad=False
        
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.L*self.C, self.C)
        self.fc2 = nn.Linear(self.L, 1)
        self.fc3 = nn.ModuleList([nn.Linear(self.L, 1) for i in range(self.C)])

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
            
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        
        x = self.conv_sim(x).squeeze() #(N, C, L)
        #x = F.softmax(x, dim=1)
        
        # flatten or chunk-share fc or chunk-no-share fc
        
        # flatten
        #x = self.fc1(self.dropout(x.view(-1, self.C*self.L)))
        
        # chunk-share fc
        #x = self.fc2(self.dropout(x)).squeeze()
        x = [self.fc2(self.dropout(i)) for i in torch.chunk(x, self.C, 1)]
        x = torch.cat(x, 1).squeeze()
        
        # chunk-no-share fc
        #x = torch.chunk(x, self.C, 1)
        #x = [self.fc3[i](self.dropout(x[i])) for i in range(self.C)]
        #x = torch.cat(x, 1).squeeze()
        return x


class SimLSTM(nn.Module):
    def __init__(self, args):
        super(SimLSTM, self).__init__()
        self.V = args.num_embeddings
        self.D = args.D
        self.C = args.C
        self.Ci = args.Ci
        self.layers = args.rnn_layers
        self.drop = args.drop if self.layers > 1 else 0
        weight_matrix = args.weight
        self.bidirectional =  args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        
        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if args.static:
            self.embed.weight.requires_grad=False
        if args.sim_static:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.D+self.C,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        #LSTM Initialization, 
        for name, params in self.rnn.named_parameters():
            #weight: Orthogonal Initialization
            if 'weight' in name:
                nn.init.orthogonal_(params)
            #lstm forget gate bias init with 1.0
            if 'bias' in name:
                b_i, b_f, b_c, b_o = params.chunk(4, 0)
                nn.init.ones_(b_f)
            
        self.dropout = nn.Dropout(args.drop)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (N, L, D)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = sim.permute(0,2,1) ##(N, L, C)
        
        # concatenate x and sim --> (N, L, D+C)
        x = torch.cat([x,sim],2)
        
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

class SimAttn(nn.Module):
    """
    sims --> attention probability distribution
    """
    def __init__(self, args):
        super(SimAttn, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.C*self.C, self.C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        #position encoding
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = F.softmax(sim,dim=2)
        x = torch.matmul(sim, x) # (N, C, D), C*L * L*D
        sim = x.unsqueeze(1)
        sim = self.dropout(self.conv_sim(sim).squeeze()) #(N, C, C)
        sim = self.fc1(sim.view(-1,self.C*self.C)) 
            
        return sim


class SimAttn1(nn.Module):
    """
    sims --> attention probability distribution
    """
    def __init__(self, args):
        super(SimAttn1, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.C*self.D, self.C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        #position encoding
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = F.softmax(sim,dim=2)
        x = torch.matmul(sim, x) # (N, C, D), C*L * L*D
        sim = self.fc1(self.dropout(x.view(-1,self.C*self.D))) 
            
        return sim


class SimAttnX(nn.Module):
    """
    sims --> attention probability distribution
    """
    def __init__(self, args):
        super(SimAttnX, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.C, self.C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        #position encoding
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = F.softmax(sim,dim=2)
        x = torch.matmul(sim, x) # (N, C, D), C*L * L*D
        sim = x.unsqueeze(1)     # (N,1,C,D)
        sim = self.conv_sim(sim).squeeze().permute(0,2,1) #(N, C, C)
        #softmax or not?
        sim = F.softmax(sim, dim=1) #not sure dim=2 or 1
        sim = 2*torch.cat([sim[:,i,i] for i in range(self.C)]).view(-1, self.C)-torch.sum(sim,dim=1)
        #sim = self.fc1(sim) 
            
        return sim


class SimAttnPE1(nn.Module):
    """
    sims --> attention probability distribution
    """
    def __init__(self, args):
        super(SimAttnPE1, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.C*self.D, self.C)
        self.pe = PositionalEncoding(self.D, 0.1)
        self.pe1 = PositionalEncoding1(self.D, 0, self.L)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        #position encoding
        #x = self.pe(x)
        x = self.pe1(x)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = F.softmax(sim,dim=2)
        x = torch.matmul(sim, x) # (N, C, D), C*L * L*D
        sim = self.fc1(self.dropout(x.view(-1,self.C*self.D))) 
            
        return sim
    

class SimAttn2(nn.Module):
    """
    sims --> attention probability distribution
    """
    def __init__(self, args):
        super(SimAttn2, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.D, self.C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        #position encoding
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = F.softmax(sim,dim=2)
        x = torch.matmul(sim, x) # (N, C, D), C*L * L*D
        #x = torch.matmul(self.dropout(sim), x) # (N, C, D), C*L * L*D
        sim = self.fc1(self.dropout(torch.mean(x,dim=1).squeeze()))
        #sim = self.fc1(torch.mean(x,dim=1).squeeze())
        #sim = torch.matmul(sim, self.conv_sim.weight)
            
        return sim
    

class SimAttn3(nn.Module):
    """
    sims --> attention probability distribution
    """
    def __init__(self, args):
        super(SimAttn3, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.D, 1)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        #position encoding
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = F.softmax(sim,dim=2)
        x = torch.matmul(sim, x) # (N, C, D), C*L * L*D
        #x = torch.matmul(self.dropout(sim), x) # (N, C, D), C*L * L*D
        #sim = self.fc1(self.dropout(torch.mean(x,dim=1).squeeze()))
        sim = self.fc1(x).squeeze()
            
        return sim
    

class SimCnnPe(nn.Module):
    """
    use Sim_CNN to learn label presentation and compute similarities to help classification
    """
    
    def __init__(self, args):
        super(SimCnnPe, self).__init__()
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
        self.pe = SinglePE(self.L)
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        self.conv_sim = nn.Conv2d(Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False
        
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.L*(self.C+1), self.C)
        self.fc2 = nn.Linear(self.L, 1)
        self.fc3 = nn.ModuleList([nn.Linear(self.L, 1) for i in range(self.C)])

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
            
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        
        x = self.conv_sim(x).squeeze() #(N, C, L)
        x = x.permute(0,2,1)
        x = self.pe(x)
       
        x = self.fc1(self.dropout(x.view(-1, (self.C+1)*self.L)))
 
        return x                

class Sim_CNN_PE(nn.Module):
    """
    use Sim_CNN to learn label presentation and compute similarities to help classification
    """
    
    def __init__(self, args):
        super(Sim_CNN_PE, self).__init__()
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


class SimLSTM1(nn.Module):
    def __init__(self, args):
        super(SimLSTM1, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
#        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.C,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            #nn.init.zeros_(self.rnn.bias_ih_l0)
            #nn.init.zeros_(self.rnn.bias_hh_l0)
            #nn.init.ones_(self.rnn.bias_ih_l0)
            #nn.init.ones_(self.rnn.bias_hh_l0)
            
        if self.layers==2:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
            #how about 2nd layer? initialization like layer0 or keep default?
            
        self.dropout = nn.Dropout(args.drop)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (N, L, D)
        
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        x = self.conv_sim(x).squeeze() #(N, C, L)
        #activate
        #x = F.relu(x)
        #x = F.tanh(x) #deprecated warning
        #x = F.leaky_relu(x)
        #x =  F.softsign(x)
        x = F.hardtanh(x)
        x = x.permute(0,2,1)
    
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


class SimLSTM2(nn.Module):
    def __init__(self, args):
        super(SimLSTM2, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_doc = nn.Conv1d(1, self.C, self.num_directions*self.hidden_size)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.C,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            #nn.init.zeros_(self.rnn.bias_ih_l0)
            #nn.init.zeros_(self.rnn.bias_hh_l0)
            #nn.init.ones_(self.rnn.bias_ih_l0)
            #nn.init.ones_(self.rnn.bias_hh_l0)
            
        if self.layers==2:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
            #how about 2nd layer? initialization like layer0 or keep default?
            
        self.dropout = nn.Dropout(args.drop)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (N, L, D)
        
        x = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        x = self.conv_sim(x).squeeze() #(N, C, L)
        #activate
        #x = F.relu(x)
        #x = F.tanh(x) #deprecated warning
        #x = F.leaky_relu(x)
        #x =  F.softsign(x)
        #x = F.hardtanh(x)
        x = x.permute(0,2,1)
    
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
        
        #out = self.fc(self.dropout(out))
        out = out.unsqueeze(1)
        out = self.conv_doc(out).squeeze()
            
        return out


class SimLSTM3(nn.Module):
    def __init__(self, args):
        super(SimLSTM3, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_doc = nn.Conv1d(1, self.C, self.num_directions*self.hidden_size)
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
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            #nn.init.zeros_(self.rnn.bias_ih_l0)
            #nn.init.zeros_(self.rnn.bias_hh_l0)
            #nn.init.ones_(self.rnn.bias_ih_l0)
            #nn.init.ones_(self.rnn.bias_hh_l0)
            
        if self.layers==2:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
            #how about 2nd layer? initialization like layer0 or keep default?
            
        self.dropout = nn.Dropout(args.drop)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.C)

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
        
        out = self.dropout(out)
        out = out.unsqueeze(1)
        out = self.conv_doc(out).squeeze()
            
        return out


class SimLSTM4(nn.Module):
    def __init__(self, args):
        super(SimLSTM4, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.D+self.C,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            #nn.init.zeros_(self.rnn.bias_ih_l0)
            #nn.init.zeros_(self.rnn.bias_hh_l0)
            #nn.init.ones_(self.rnn.bias_ih_l0)
            #nn.init.ones_(self.rnn.bias_hh_l0)
            
        self.dropout = nn.Dropout(args.drop)
        self.fc = nn.Linear(self.num_directions*self.hidden_size+self.D+self.C, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (B, L, D)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = sim.permute(0,2,1) ##(N, L, C)
        
        # concatenate x and sim --> (N, L, D+C)
        x = torch.cat([x,sim],2)
        
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
        
        #concatenate sum/avg embeddings
        #out = torch.cat([out,torch.sum(x,dim=1).squeeze()],dim=1)
        out = torch.cat([out,torch.mean(x,dim=1).squeeze()],dim=1)
        out = self.fc(self.dropout(out))
            
        return out


class SimLSTM5(nn.Module):
    def __init__(self, args):
        super(SimLSTM5, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.D+self.C,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
            #nn.init.zeros_(self.rnn.bias_ih_l0)
            #nn.init.zeros_(self.rnn.bias_hh_l0)
            #nn.init.ones_(self.rnn.bias_ih_l0)
            #nn.init.ones_(self.rnn.bias_hh_l0)
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.D+self.C, self.num_directions*self.hidden_size)
        self.fc2 = nn.Linear(self.num_directions*self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        #(h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (B, L, D)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = sim.permute(0,2,1) ##(N, L, C)
        
        # concatenate x and sim --> (N, L, D+C)
        x = torch.cat([x,sim],2)
        
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
        
        #add sum/avg embeddings
        #out = out + self.fc1(self.dropout(torch.sum(x,dim=1).squeeze()))
        out = out + self.fc1(self.dropout(torch.mean(x,dim=1).squeeze()))
        
        out = self.fc2(self.dropout(out))
            
        return out


class SimLSTM6(nn.Module):
    def __init__(self, args):
        super(SimLSTM6, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        if static == 1:
            self.embed.weight.requires_grad=False
            self.conv_sim.weight.requires_grad=False
        elif static == 2:
            self.embed.weight.requires_grad=False
        elif static == 3:
            self.conv_sim.weight.requires_grad=False

        self.rnn = nn.LSTM(               
            input_size=self.D+self.C,                #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=self.bidirectional  #bi-LSTM
        )
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.num_directions*self.hidden_size, self.D+self.C)
        self.fc2 = nn.Linear(self.D+self.C, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = sim.permute(0,2,1) ##(N, L, C)
        
        # concatenate x and sim --> (N, L, D+C)
        x = torch.cat([x,sim],2)
        
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
        
        #add sum/avg embeddings
        out = self.fc1(self.dropout(out)) + torch.mean(x,dim=1).squeeze()
        
        out = self.fc2(self.dropout(out))
            
        return out


class SimLSTM7(nn.Module):
    def __init__(self, args):
        super(SimLSTM7, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
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
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.num_directions*self.hidden_size+self.C, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = sim.permute(0,2,1) ##(N, L, C)
        
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose all time steps' output
        r_out = r_out #lstm/bilstm 
        
        #concatenate sims
        r_out = torch.cat([r_out,sim],2)
        
        r_out = self.fc1(self.dropout(torch.mean(r_out,dim=1).squeeze()))
            
        return r_out


class SimLSTM8(nn.Module):
    def __init__(self, args):
        super(SimLSTM8, self).__init__()
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
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
            input_size=self.hidden_size*self.num_directions+self.C, #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=False               #LSTM
        )
        
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.hidden_size, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        sim = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim = self.conv_sim(sim).squeeze() #(N, C, L)
        sim = sim.permute(0,2,1) #(N, L, C)
        
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose all time steps' output, i.e. r_out
        
        #concatenate sims
        r_out = torch.cat([r_out,sim],2)
        
        #lstm, choose last time step 
        r_out, (h_n, c_n) = self.lstm(r_out, None)
        r_out = r_out[:,-1,:]
        r_out = self.fc1(self.dropout(r_out))
            
        return r_out


class SimLSTM9(nn.Module):
    def __init__(self, args):
        super(SimLSTM9, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
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
            input_size=self.C, #The number of expected features in the input x 
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=False               #LSTM
        )
        
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.hidden_size, self.C)
        self.bilinear = nn.Bilinear(self.C, self.C, self.C, bias=True)
        #nn.Bilinear without bias broken #5990, it a bug
        self.fc2 = nn.Linear(self.L*self.C, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        sim1 = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim1 = self.conv_sim(sim1).squeeze() #(N, C, L)
        sim1 = sim1.permute(0,2,1).contiguous() #(N, L, C)
        
        #initialization hidden state
        #1.zero init
        x, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose all time steps' output, i.e. r_out
        
        #compute sim2
        sim2 = x.unsqueeze(1)  # (N, Ci, L, self.num_directions*self.hidden_size)
        sim2 = self.conv_sim2(sim2).squeeze() #(N, C, L)
        sim2 = sim2.permute(0,2,1).contiguous() #(N, L, C)
        #make sure sim,sim2 are contiguous
        #print(sim1.is_contiguous(),sim2.is_contiguous())
        
        #use bilinear transform to weight sim1 and sim2
        w = torch.sigmoid(self.bilinear(sim1, sim2))
        
        #merge the 2 sims: concatenate, sum, weighted-sum, bilinear transform...
        sim = w.mul(sim1)+(torch.ones_like(w)-w).mul(sim2)
        
        #lstm, choose last time step 
        sim, (h_n, c_n) = self.lstm(sim, None)
        sim = sim[:,-1,:]
        sim = self.fc1(self.dropout(sim))
        
        #sim = self.fc2(self.dropout(sim.view(-1,self.L*self.C)))
            
        return sim


class SimLSTM10(nn.Module):
    def __init__(self, args):
        super(SimLSTM10, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
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
            input_size=self.C*2, #The number of expected features in the input x 
            hidden_size=self.C,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=False               #LSTM
        )
        
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(2*self.C*self.L, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        sim1 = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim1 = self.conv_sim(sim1).squeeze() #(N, C, L)
        sim1 = sim1.permute(0,2,1) #(N, L, C)
        
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose all time steps' output, i.e. r_out
        
        #compute sim2
        sim2 = r_out.unsqueeze(1)  # (N, Ci, L, self.num_directions*self.hidden_size)
        sim2 = self.conv_sim2(sim2).squeeze() #(N, C, L)
        sim2 = sim2.permute(0,2,1) #(N, L, C)
        
        #merge the 2 sims: concatenate, sum, weighted-sum, bilinear transform...
        sim = torch.cat([sim1,sim2],dim=2)
        
#        #lstm, choose last time step 
#        sim, (h_n, c_n) = self.lstm(sim, None)
#        sim = sim[:,-1,:]
        
        sim = self.fc1(self.dropout(sim.view(-1, 2*self.C*self.L)))
            
        return sim


class SimLSTM11(nn.Module):
    def __init__(self, args):
        super(SimLSTM11, self).__init__()
        self.V = args.num_embeddings
        self.L = args.L
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
        self.conv_sim = nn.Conv2d(self.Ci, self.C, (1, self.D)) #10 label vec kernel with size(1,D)
        self.conv_sim.weight = torch.nn.Parameter(args.label_vecs)
        self.conv_sim2 = nn.Conv2d(self.Ci, self.C, (1, self.num_directions*self.hidden_size))
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
            input_size=self.C*2, #The number of expected features in the input x 
            hidden_size=self.C,     # rnn hidden unit
            num_layers=self.layers,           # number of rnn layers
            batch_first=True,                 # set batch first
            dropout=self.drop,                #dropout probability
            bidirectional=False               #LSTM
        )
        
        #pytorch中rnn/lstm/gru权重和偏置默认都是均匀初始化的，一般要将权重改为正交初始化，LSTM的forget gate的bias初始化为1
        #Orthogonal Initialization, 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用
        if self.layers==1:
            #weight: Orthogonal Initialization
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            #bias: zero init or 1 init; how to set LSTM's forget gate's bias to 1?
           
            
        self.dropout = nn.Dropout(args.drop)
        self.fc1 = nn.Linear(self.C*self.L, self.C)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        
        sim1 = x.unsqueeze(1)  # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        sim1 = self.conv_sim(sim1).squeeze() #(N, C, L)
        sim1 = sim1.permute(0,2,1) #(N, L, C)
        sim1 = F.softmax(sim1,dim=2)
        
        #initialization hidden state
        #1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        
        # choose all time steps' output, i.e. r_out
        
        #compute sim2
        sim2 = r_out.unsqueeze(1)  # (N, Ci, L, self.num_directions*self.hidden_size)
        sim2 = self.conv_sim2(sim2).squeeze() #(N, C, L)
        sim2 = sim2.permute(0,2,1) #(N, L, C)
        sim2 = F.softmax(sim2,dim=2)
        
        #merge the 2 sims: concatenate, sum, weighted-sum, bilinear transform...
        sim = F.softmax((sim1+sim2)/2,dim=2)
        
#        #lstm, choose last time step 
#        sim, (h_n, c_n) = self.lstm(sim, None)
#        sim = sim[:,-1,:]
        
        sim = self.fc1(self.dropout(sim.view(-1, self.C*self.L)))
            
        return sim
    
    
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
        x = x + (self.pe[:, :x.size(1)]).clone().detach()#词嵌入+位置嵌入
        return self.dropout(x)


class PositionalEncoding1(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding1, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) #max_len: max doc length
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.tensor(1/max_len)
        #all dimentions share the same position encoding
        pe[:] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)
        self.coef = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = (1-self.coef)*x + self.coef*(self.pe[:, :x.size(1)]).clone().detach()#词嵌入+位置嵌入
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

class SinglePE(nn.Module):
    "a simple PE version without considering dimention"
    def __init__(self, L):
        super(SinglePE, self).__init__()
        self.L = L
        
    
    def forward(self, x):
        #div_term:1, L^0.5, L, len(vocab)?
        x = torch.cat((x, torch.tensor([[torch.sin(torch.tensor(i/self.L)) for i in range(self.L)] for j in range(x.size(0))]).unsqueeze(dim=2).cuda()),dim=2) 
        return x
        
