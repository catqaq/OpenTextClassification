#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:13:19 2019

@author: jjg
"""

import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import copy
from torchtext import data
from torchtext.vocab import Vectors
from nltk.tokenize import word_tokenize
from models import TextCNN, LSTM, GRU, Hybrid_CNN
from sim_cnn import Sim_CNN
from settings import Settings
from train_eval import training, evaluating

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '/home/jjg/data/text_classification/dbpedia'
class_vecs = torch.from_numpy(np.load(data_path+'/classname_based_class_vecs.npy'))
length=100

cache = 'mycache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(
    name='googlenews.txt', #or other pre-trained word vector file in txt format 
    cache=cache)  

############################### construct vocab ##################

#we've finished tokenization above，set tokenize=None, better set batch_first=True
TEXT = data.Field(
    sequential=True,
    tokenize=word_tokenize,
    lower=True,
    fix_length=length,    #according to the dataset
    batch_first=True)
  
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)

train, test = data.TabularDataset.splits(
        path=data_path,
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=[('label', LABEL), ('text', TEXT)],
        skip_header=True)

#split validation set
train, dev = train.split(split_ratio=0.7)

#construct the vocab
#TEXT.build_vocab(train, vectors="glove.840B.300d")
TEXT.build_vocab(train, vectors=vectors)
vocab = TEXT.vocab
del vectors               #del vectors to save space


#can choose 3 kinds of models
models = {'CNN': TextCNN, 'LSTM': LSTM, 'GRU': GRU, 'Sim_CNN': Sim_CNN, 'Hybrid': Hybrid_CNN}

if __name__ == '__main__':
    #settings
    #classifier='Sim_CNN'         #choose CNN/LSTM/GRU
    #see settings.py for detail
    label_vecs =class_vecs.unsqueeze(1).unsqueeze(2)
    args = Settings(
        vocab.vectors,            #pre-trained word embeddings
        label_vecs,
        L=length,
        Dim=300,                       #embedding dimension
        num_class=14,
        Cout=256,                       #kernel numbers
        kernel_size=[3, 4, 5],         #different kernel size
        dropout=0.5,
        batch_size=256,
        num_epochs=10,
        lr=0.0001,
        weight_decay=0,
        static=4,                   #update the embeddings or not
        batch_normalization=False,
        hidden_size=100,
        rnn_layers=2,
        bidirectional=False)
    
    #construct dataset iterator
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
           (train, dev, test),
           sort_key=lambda x: len(x.text), 
           batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
           )
    
    res = []
    for classifier in ['CNN']:
        
        if args.static:
            print('static %s(without updating embeddings):' % classifier)
        else:
            print('non-static %s(update embeddings):' % classifier)
        
        dev_acc = []
        t = 1  #repeat times
        best_acc = 0
        for i in range(t):
            model = models[classifier](args)
            training(train_iter, dev_iter, model, args, device)
            tmp = evaluating(dev_iter, model, device)
            dev_acc.append(tmp)
            if tmp > best_acc:
                best_acc = tmp
                best_model_wts = copy.deepcopy(model.state_dict())
        print('Repeat %s times: %s' % (t, dev_acc))
        print('average dev_acc: %.1f%%' % (100*sum(dev_acc)/t))
        print('max dev_acc: %.1f%%' % (100*best_acc))
        model.load_state_dict(best_model_wts)
        #finally evaluate the test set
        test_acc = evaluating(test_iter, model, device)
        print('test acc: %.1f%%' % (test_acc*100))
        res.append(test_acc)
    print(res)
    