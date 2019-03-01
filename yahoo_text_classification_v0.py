#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:36:25 2019

@author: jjg
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

 #hyper-parameters
input_size = 300
sims_input_size = 310   #300+10
sim_factor = 1      #Similarity factor
num_classes = 10
num_epochs = 200
batch_size = 128
learning_rate = 0.00001

#loading data
train_x=np.load('yahoo90000_train_vec.npy')
train_y=np.load('yahoo90000_train_label.npy')

test_x=np.load('yahoo_test_vec.npy')
test_y=np.load('yahoo_test_label.npy')

class_vecs=np.load('yahoo_class_vec_based_on_classname.npy')

#add similarities
def add_similarities(doc_vec, class_vec_list):
    sims = []
    for class_vec in class_vec_list:
        try:
            sims.append(dot(doc_vec,class_vec)/(norm(doc_vec)*norm(class_vec)))
        except:
            pass
    return np.append(doc_vec,np.array(sims)*sim_factor)

#compute vec_with_sims
train_sims_x=np.array([add_similarities(train_x[i],class_vecs) for i in range(len(train_x))])
test_sims_x=np.array([add_similarities(test_x[i],class_vecs) for i in range(len(test_x))])

#change ndarray to tensor
train_x = torch.from_numpy(train_x)
test_x = torch.from_numpy(test_x)

train_sims_x=torch.from_numpy(train_sims_x)
test_sims_x=torch.from_numpy(test_sims_x)

train_y = torch.from_numpy(train_y)
test_y = torch.from_numpy(test_y)

#note:Pytorch has a requirement that the label must start with 0 
#when using the CrossEntropyLoss function.
train_y = train_y-1
test_y = test_y-1

#yahoo_dataset
train_dataset = Data.TensorDataset(train_x, train_y)
test_dataset = Data.TensorDataset(test_x, test_y)

train_sims_dataset = Data.TensorDataset(train_sims_x, train_y)
test_sims_dataset = Data.TensorDataset(test_sims_x, test_y)

#Data loader
train_loader = Data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_sims_loader = Data.DataLoader(
    dataset=train_sims_dataset, batch_size=batch_size, shuffle=True)
test_sims_loader = Data.DataLoader(
    dataset=test_sims_dataset, batch_size=batch_size, shuffle=False)

#Logistic regression model
#Note: the input dimensions are different
model = nn.Linear(input_size, num_classes)  #linear transformation

sims_model = nn.Linear(sims_input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

sims_criterion = nn.CrossEntropyLoss()
sims_optimizer = torch.optim.SGD(sims_model.parameters(), lr=learning_rate)
#sims_optimizer = torch.optim.Adam(sims_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

#Train the model with 300-d text vector
total_step = len(train_loader)  #how many batches?
for epoch in range(num_epochs):
    for i, (vecs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(vecs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# =============================================================================
#         #Visualization of the training process
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
#                 epoch + 1, num_epochs, i + 1, total_step, loss.item()))
# 
# =============================================================================

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for vecs, labels in test_loader:
        outputs = model(vecs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy(300-d text vector): %.2f%%' %
          (100 * correct.item() / total))

#Train the sims_model with 310-d vector
sims_total_step = len(train_sims_loader)  #how many batches?
for epoch in range(num_epochs):
    for i, (vecs, labels) in enumerate(train_sims_loader):
        # Forward pass
        outputs = sims_model(vecs)
        loss = sims_criterion(outputs, labels)

        # Backward and optimize
        sims_optimizer.zero_grad()
        loss.backward()
        sims_optimizer.step()
# =============================================================================
#       #Visualization of the training process
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
#                 epoch + 1, num_epochs, i + 1, sims_total_step, loss.item()))
# =============================================================================

with torch.no_grad():
    correct = 0
    total = 0
    for vecs, labels in test_sims_loader:
        outputs = sims_model(vecs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy(310-d vector with similarities): %.2f%%' %
          (100 * correct.item() / total))
    