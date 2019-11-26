#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:34:18 2019

@author: jjg
"""

import pandas as pd
from nltk import word_tokenize
import os

task = '/home/jjg/data/text_classification'
data = 'yahoo'
data_path = os.path.join(task, data)
#raw: raw data
raw_path = os.path.join(task, data, 'raw')

#load data
full = pd.read_csv(os.path.join(raw_path, 'train.csv'), header=None, sep=',')

test = pd.read_csv(os.path.join(raw_path, 'test.csv'), header=None, sep=',')


#concatenate all columns except the label column
def preprocess(dataset):
    dataset.fillna('', inplace=True)
    dataset['text'] = dataset[1].str.cat(
        [dataset[i] for i in range(2, dataset.shape[1])], sep=' ')
    dataset.drop(columns=range(1, dataset.shape[1] - 1), inplace=True)
    return dataset


full = preprocess(full)
test = preprocess(test)

full.rename(columns={0: 'label'}, inplace=True)
test.rename(columns={0: 'label'}, inplace=True)

#split train and test (or u can use k-fold Cross-validation)
train = full.sample(frac=0.9, random_state=0, axis=0)
dev = full[~full.index.isin(train.index)]

train.to_csv(os.path.join(data_path, 'train.csv'), header=True, index=False)

dev.to_csv(os.path.join(data_path, 'dev.csv'), header=True, index=False)

test.to_csv(os.path.join(data_path, 'test.csv'), header=True, index=False)


#get text length
def get_long(series):
    return series.apply(lambda x: len(word_tokenize(str(x))))


train['long'] = get_long(train['text'])
test['long'] = get_long(test['text'])

train['long'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
#dbpedia:
#count    504000.000000
#mean         53.722917
#std          24.564807
#min           3.000000
#25%          33.000000
#50%          54.000000
#75%          74.000000
#90%          86.000000
#95%          90.000000
#99%          98.000000
#max        1499.000000

#yahoo:
#count    1.260000e+06
#mean     1.070280e+02
#std      1.150473e+02
#min      2.000000e+00
#25%      3.700000e+01
#50%      7.100000e+01
#75%      1.330000e+02
#90%      2.310000e+02
#95%      3.160000e+02
#99%      6.300000e+02
#max      3.997000e+03