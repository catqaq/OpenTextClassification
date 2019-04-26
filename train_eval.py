#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:53:31 2019

@author: jjg
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     
def train(train_iter, val_iter, model, args, device):
    l2 = args.l2
    static = args.static
    model.to(device)  #move model to device before constructing optimizer for it.
    if static:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=l2)
        #optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=l2)
        #optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=l2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        #optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    total_step = len(train_iter)
    train_acc = []
    val_acc = []
    t0 = time()
    for epoch in range(1, args.epochs + 1):
        model.train()  #training mode, we should reset it to training mode in each epoch
        for i, batch in enumerate(train_iter):
            texts, labels = batch.full.to(device), batch.label.to(device) - 1
            optimizer.zero_grad()
            outputs = model(texts)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            #Visualization of the train process
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch, args.epochs, i+1, total_step, loss.item()))
    #compute train accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in train_iter:
                texts, labels = batch.full.to(
                    device), batch.label.to(device) - 1
                outputs = model(texts)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum()
            train_acc.append(correct.item() / total)
        #in each epoch we call eval(), switch to evaluation mode to compute test_acc
        val_acc.append(eval(val_iter, model, device)) 
    t1 = time()
    print('training time: %.2f' % (t1 - t0))

    #plot train acc and test acc
    plt.ion()
    plt.figure()
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.title('Training and validation acc')
    plt.tight_layout()
    #plt.xticks(range(0,args.epochs),range(1,args.epochs+1))
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()


def eval(data_iter, model, device):
    model.to(device)
    model.eval()  #evaluation mode
    with torch.no_grad():
        correct, avg_loss = 0, 0
        for batch in data_iter:
            texts, labels = batch.full.to(device), batch.label.to(device) - 1

            outputs = model(texts)
            predicted = torch.max(outputs.data, 1)[1]
            loss = F.cross_entropy(outputs, labels, reduction='mean')

            avg_loss += loss.item()
            correct += (predicted == labels).sum()

        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = correct.item() / size
        #print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 100*accuracy, correct, size))
        return accuracy