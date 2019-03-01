#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:03:59 2019

@author: jjg
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from time import time

# 1.optimize hyper-parameters

# (1)Processing data, building models
input_size = 300
sims_input_size = 310  #300+10
sim_factor = 5 #Similarity factor
num_classes = 10

#hyper-parameters
num_epochs = 500
batch_size = 128
learning_rate = 0.00001

#loading data
train_x = np.load('yahoo90000_train_vec.npy')
train_y = np.load('yahoo90000_train_label.npy')

test_x = np.load('yahoo_test_vec.npy')
test_y = np.load('yahoo_test_label.npy')

class_vecs = np.load('yahoo_class_vec_based_on_classname.npy')


#add similarities
def add_similarities(doc_vec, class_vec_list):
    sims = []
    for class_vec in class_vec_list:
        try:
            sims.append(
                dot(doc_vec, class_vec) / (norm(doc_vec) * norm(class_vec)))
        except:
            pass
    return np.append(doc_vec, np.array(sims) * sim_factor)


#compute vec_with_sims
train_sims_x = np.array(
    [add_similarities(train_x[i], class_vecs) for i in range(len(train_x))])
test_sims_x = np.array(
    [add_similarities(test_x[i], class_vecs) for i in range(len(test_x))])

#change ndarray to tensor
train_x = torch.from_numpy(train_x)
test_x = torch.from_numpy(test_x)

train_sims_x = torch.from_numpy(train_sims_x)
test_sims_x = torch.from_numpy(test_sims_x)

train_y = torch.from_numpy(train_y)
test_y = torch.from_numpy(test_y)

#note:Pytorch has a requirement that the label must start with 0
#when using the CrossEntropyLoss function.
train_y = train_y - 1
test_y = test_y - 1

#yahoo_dataset
train_dataset = Data.TensorDataset(train_x, train_y)
test_dataset = Data.TensorDataset(test_x, test_y)

train_sims_dataset = Data.TensorDataset(train_sims_x, train_y)
test_sims_dataset = Data.TensorDataset(test_sims_x, test_y)

#define the subset 为了方便抽样要不要弄成个dataframe?
#Data.ConcatDataset([])
#train_subset=Data.Subset(train_dataset,indices=)

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
model = nn.Linear(input_size,
                  num_classes)  #linear transformation only,要不要弄个复杂点的网络？

sims_model = nn.Linear(sims_input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

sims_criterion = nn.CrossEntropyLoss()
#sims_optimizer = torch.optim.SGD(sims_model.parameters(), lr=learning_rate)
sims_optimizer = torch.optim.Adam(sims_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

#(2)move model and data to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
sims_model.to(device)

#(3)300_d hyper-parameters optimization
#为了优化参数，每轮训练过后都求一遍train_accuracy和test_accuracy
#train_losses = [] #训练集每轮中最后一批的误差,波动性很大，换用train_accuracy
#test_losses = []  #测试集每轮中最后一批的误差(不太全面)

train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    for i, (vecs, labels) in enumerate(train_loader):
        # Forward pass
        vecs, labels = vecs.to(device), labels.to(device)
        outputs = model(vecs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #compute train accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for vecs, labels in train_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            outputs = model(vecs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        train_accuracy.append(correct.item() / total)

    #     with torch.no_grad():
    #         for vecs, labels in test_loader:
    #             outputs = model(vecs)
    #             test_loss = criterion(outputs, labels)
    #         test_losses.append(test_loss)

    with torch.no_grad():
        correct = 0
        total = 0
        for vecs, labels in test_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            outputs = model(vecs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_accuracy.append(correct.item() / total)

# print('GPU 100 epochs: %.2f' % (t1 - t0))#135.23
# print('CPU 100 epochs: %.2f' % (t1 - t0))# 130.53 cpu反而稍快一点点，可能是因为模型很简单，数据量又小，体现不出GPU的优势

# print('GPU 500 epochs: %.2f' % (t1 - t0)) #661.90
# print('CPU 500 epochs: %.2f' % (t1 - t0))# 642.46

#(4)310_d hyper-parameters optimization       
train_sims_accuracy = []
test_sims_accuracy = []

for epoch in range(num_epochs):
    for i, (vecs, labels) in enumerate(train_sims_loader):
        # Forward pass
        vecs, labels = vecs.to(device), labels.to(device)
        outputs = sims_model(vecs)
        loss = sims_criterion(outputs, labels)

#         #compute train accuracy
#         _, predicted = torch.max(outputs.data, 1)#这里用的是每批训练完之后的model，outputs = model(vecs)，应该是训练完一轮之后再求outputs
#         total += labels.size(0)
#         correct += (predicted == labels).sum()

        # Backward and optimize
        sims_optimizer.zero_grad()
        loss.backward()
        sims_optimizer.step()
        
    #compute train accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for vecs, labels in train_sims_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            outputs = sims_model(vecs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        train_sims_accuracy.append(correct.item() / total)

    with torch.no_grad():
        correct = 0
        total = 0
        for vecs, labels in test_sims_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            outputs = sims_model(vecs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_sims_accuracy.append(correct.item() / total)

#(5)plot the accuracy-epoch curve
plt.ylabel('Accuracy')
plt.xlabel('Training epochs')
plt.title('Impact of epochs on train accuracy and test accuracy')
plt.tight_layout()
plt.plot(train_accuracy, label='300d_train_accuracy')
plt.plot(test_accuracy, label='300d_test_accuracy')
plt.plot(train_sims_accuracy, label='310d_train_accuracy')
plt.plot(test_sims_accuracy, label='310d_test_accuracy')
plt.legend()

#只画出部分
plt.ylabel('Accuracy')
plt.xlabel('Training epochs')
plt.title('Impact of epochs on train accuracy and test accuracy')
plt.tight_layout()
plt.plot(np.arange(200,num_epochs),train_accuracy[200:], label='300d_train_accuracy')
plt.plot(np.arange(200,num_epochs),test_accuracy[200:], label='300d_test_accuracy')
plt.plot(np.arange(200,num_epochs),train_sims_accuracy[200:], label='310d_train_accuracy')
plt.plot(np.arange(200,num_epochs),test_sims_accuracy[200:], label='310d_test_accuracy')
plt.legend()

