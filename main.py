#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:13:19 2019

@author: jjg
"""

import pandas as pd
import numpy as np
import torch
import os
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
from nltk.tokenize import word_tokenize
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models import TextCNN, LSTM, GRU, Sim_CNN
from settings import Settings
from train_eval import train, eval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load data
train_set = pd.read_csv(
    '../yahoo_answers_csv/train.csv',
    header=None,
    sep=',',
    names=['label', 'question_title', 'question_content', 'answer'])

test_set = pd.read_csv(
    '../yahoo_answers_csv/test.csv',
    header=None,
    sep=',',
    names=['label', 'question_title', 'question_content', 'answer'])

class_vecs = torch.from_numpy(np.load('../yahoo_class_vec_based_on_classname.npy'))

#subset: 1000 samples/ class
n = 1000
train_set = pd.concat(
    [train_set[train_set.label == i][:n] for i in range(1, 11)],
    ignore_index=True)


def preprocess(dataset):
    dataset.fillna('', inplace=True)
    dataset['full'] = dataset['question_title'].str.cat(
        [dataset['question_content'], dataset['answer']], sep=' ')
    dataset.drop(
        columns=['question_title', 'question_content', 'answer'], inplace=True)
    dataset['full'] = dataset['full'].apply(lambda x: word_tokenize(str(x)))
    return dataset


train_set = preprocess(train_set)
test_set = preprocess(test_set)

cache = 'mycache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(
    name='googlenews.txt', #or other pre-trained word vector file in txt format 
    cache=cache)  

############################### construct vocab ##################

#we've finished tokenization above，set tokenize=None, better set batch_first=True
TEXT = data.Field(sequential=True, lower=True, fix_length=300, batch_first=True)
  
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
TEXT.build_vocab(
    train_set['full'],
    test_set['full'],
    vectors=vectors,
    unk_init=init.xavier_uniform_)

del vectors               #del vectors to save space


# construct examples and fields
def get_dataset(csv_data, text_field, label_field):

    fields = [("full", text_field), ("label", label_field)]
    examples = []
    for text, label in tqdm(zip(csv_data['full'], csv_data['label'])):
        examples.append(data.Example.fromlist([text, label], fields))
    return examples, fields


train_examples, train_fields = get_dataset(train_set, TEXT, LABEL)
test_examples, test_fields = get_dataset(test_set, TEXT, LABEL)

# construct dataset
train_dataset = data.Dataset(train_examples, train_fields)
test_dataset = data.Dataset(test_examples, test_fields)

#split validation set
train_dataset, val_dataset = train_dataset.split(split_ratio=0.7)

#can choose 3 kinds of models
models = {'CNN': TextCNN, 'LSTM': LSTM, 'GRU': GRU, 'Sim_CNN': Sim_CNN}

if __name__ == '__main__':
    #settings
    classifier='CNN'         #choose CNN/LSTM/GRU
    #see settings.py for detail
    label_vecs =class_vecs.unsqueeze(1).unsqueeze(2)
    args = Settings(
        TEXT.vocab.vectors,
        label_vecs,
        Dim=300,
        Cout=64,
        kernel_size=[3, 4, 5],
        dropout=0.5,
        batch_size=64,
        num_epochs=10,
        lr=0.001,
        weight_decay=0,
        static=True,
        batch_normalization=False,
        hidden_size=100,
        rnn_layers=2,
        bidirectional=False)
    
    #construct dataset iterator
    train_iter, val_iter = data.BucketIterator.splits(
        datasets=(train_dataset, val_dataset),
        batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.full),
        shuffle=True,
        sort_within_batch=False,
        repeat=False)
    test_iter = data.Iterator(
        test_dataset,
        batch_size=args.batch_size,
        sort=False,
        sort_within_batch=False,
        repeat=False)

    if args.static:
        print('static %s(without updating embeddings):' % classifier)
    else:
        print('non-static %s(update embeddings):' % classifier)
    
    test_acc = []
    t = 5  #repeat times
    max_acc = 0
    for i in range(t):
        model = models[classifier](args)
        train(train_iter, val_iter, model, args, device)
        tmp = eval(test_iter, model, device)
        test_acc.append(tmp)
        if tmp > max_acc:
            max_acc = tmp
            torch.save(model.state_dict(), './result/'+classifier+'.pt')
    print('Repeat %s times: %s' % (t, test_acc))
    print('average test_acc: %.1f%%' % (100*sum(test_acc)/t))
    print('max acc: %.1f%%' % (100*max_acc))
    
    #load the trained model
    model = models[classifier](args)
    model.load_state_dict(torch.load('./result/'+classifier+'.pt'))
    model.eval()
    print(eval(test_iter, model, device))
    
