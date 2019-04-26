import torch
import torch.nn.functional as F

#4 ways to introduce similarities to CNN's forward function

#1.r1+r2, r1 shape: (N, C), r2 shape:(N, C)
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




#2.r1+r2, and use fc3 to weight r2, r1 shape: (N, C), r2 shape:(N, C)
def forward2(self, x):
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
        r2 = self.fc3(r2)
        return r1+r2

#3.(r1;r2), r1 shape: (N, C), r2 shape:(N, C)
def forward3(self, x):
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
        #r2 = F.relu(r2).squeeze(3)
        r2 = r2.squeeze(3)
        r2 = F.avg_pool1d(r2, r2.size(2)).squeeze(2) #(N, 10)
        #r2 = F.max_pool1d(r2, r2.size(2)).squeeze(2)
        r1 = torch.cat([r1, r2], 1) #(N, 20)
        r1 = self.dropout(r1)
        r1 = self.fc2(r1)
        return r1

#4.(r1;r2), r1 shape: (N, len(Ks)*Co), r2 shape:(N, C)
def forward4(self, x):
        x = self.embed(x)  # (N, W, D)
        
        x = x.unsqueeze(1)  # (N, Ci, W, D), insert a dimention of size one(in_channels Ci)
        
        if self.use_bn:
            x=self.bn2d(x)
        
        #random initialization kernel + label vector initialization kernel
        #random kernel
        r1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, W), ...]*len(Ks)

        r1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in r1]  # [(N, Co), ...]*len(Ks)
        
        #label vector kernel
        r2 = self.conv_sim(x) #(N, 10, W)
        #r2 = F.relu(r2).squeeze(3)
        r2 = r2.squeeze(3)
        #r2 = F.max_pool1d(r2, r2.size(2)).squeeze(2)
        r2 = F.avg_pool1d(r2, r2.size(2)).squeeze(2) #(N, 10)
        
        r1 = torch.cat(r1, 1)   #(N, len(Ks)*Cout)
        
        r1 = torch.cat([r1, r2], 1) #(N, len(Ks)*Cout+C)

        r1 = self.dropout(r1)  # (N, len(Ks)*Co)
        
        r1 = self.fc4(r1)  # (N, C)

        return r1


#add similarities out of CNN
#def add_similarities(vec, class_vec_list):
#    sims = []
#    for class_vec in class_vec_list:
#        if torch.equal(vec, torch.zeros(300)):
#            sims = [0.]*10 #use 0. instead of 0
#        else:
#            sims.append(
#                torch.dot(vec, class_vec) / (torch.norm(vec) * torch.norm(class_vec)))
#    sims = torch.tensor(sims)       
#    return torch.cat([vec, sims])
#
#
#sim_weights = torch.cat([add_similarities(i, class_vecs) for i in TEXT.vocab.vectors], 0).view(-1, 310)