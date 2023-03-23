# coding: UTF-8
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

from tools.losses import cross_entropy_multi_label, multi_label_circle_loss
from tools.train_eval import train, init_network
from importlib import import_module
import argparse

# 基本参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
seed = 1

# 1.命令行传参：出于简洁性考虑，刻意控制了命令行参数的数量，可根据需要修改
parser = argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--lng', type=str, default="cn",
                    help='choose language: en for English, cn for Chinese, multi for multi-language')
parser.add_argument('--architecture', type=str, default="DNN",
                    help='choose model architecture: shallow for traditional method, DNN for deep neural networks, transformers for bert-type pretrained models')
parser.add_argument('--model', type=str, default="TextCNN",
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer, Bert, ERNIE')
parser.add_argument('--embedding', default='random', type=str, help='random, inner, outer')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--pretrained_name_or_path', type=str, default="cn",
                    help='pretrained model/word vector name or path')
args = parser.parse_args()

# 2.直接在代码中指定参数：提供了DNN和transformers的示例
# 数据
data_path = "./data/"
dataset = 'THUCNews'  # 数据集
sep = "\t"
# dataset = 'dbpedia'  # 数据集
# sep = ","
# dataset = 'nlp-with-disaster-tweets'  # 数据集
# sep = "\t"
data_path = "./data/"
dataset = 'rcv1'  # 数据集
sep = "\t"
dataset = data_path + dataset

# transformers类模型
n_samples = None  # None for full data, n for debug or big file
args.lng = "en"  # en, cn, multi
args.architecture = "transformers"
args.model = "bert"  # 名字需要与相应的module即py文件对齐
# args.model = "ERNIE"  # 名字需要与相应的module即py文件对齐
args.embedding = "inner"  # embedding下载需要时间
# args.pretrained_name_or_path = "bert-base-chinese"
# args.pretrained_name_or_path = "bert-base-uncased"
args.pretrained_name_or_path = "bert-base-cased"
# args.pretrained_name_or_path = "nghuyong/ernie-1.0-base-zh"
# args.pretrained_name_or_path = "swtx/ernie-2.0-base-chinese"
# args.pretrained_name_or_path = "nghuyong/ernie-3.0-base-zh"
# args.pretrained_name_or_path = "nghuyong/ernie-2.0-base-en"

# DNN类模型
# n_samples = None
# args.lng = "en"  # en, cn, multi
# args.architecture = "DNN"
# args.model = "TextCNN"  # 名字需要与相应的module即py文件对齐
# args.embedding = "outer"  # embedding下载需要时间
# args.word = True
# en word vectors
# args.pretrained_name_or_path = "glove.840B.300d.txt"
# args.pretrained_name_or_path = "wiki-news-300d-1M.vec"  # fasttext
# cn word vectors
# args.pretrained_name_or_path = "sgns.sogounews.bigram-char"
# args.pretrained_name_or_path = "cc.zh.300.vec"  # fasttext

# 超参数：大部分参数在config中设置
lr = 2e-5
batch_size = 128  # 默认128
multi_label = True
loss_fn = cross_entropy_multi_label  # multi-hot ce
loss_fn = F.binary_cross_entropy_with_logits  # bce
loss_fn = multi_label_circle_loss  # 苏剑林版多标签分类loss
eval_activate = None
cls_threshold = 0
epochs = 20

if __name__ == '__main__':
    embedding = 'embedding_SougouNews.npz' if args.lng == "cn" else "glove.840B.300d.txt"
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    if model_name == 'FastText':
        from tools.utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'
    elif args.architecture == "transformers":
        from tools.utils_transformers import build_dataset, build_iterator, get_time_dif
    else:
        from tools.utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    if args.embedding == "inner":
        # 预训练模型使用自带的embeddings
        config = x.Config(dataset, args.pretrained_name_or_path)
    else:
        config = x.Config(dataset, args.pretrained_name_or_path)
    # single/multi label
    config.multi_label = multi_label
    # 超参数设置
    config.learning_rate = lr
    config.batch_size = batch_size
    config.num_epochs = epochs
    # 保存路径+时间
    config.save_path = config.save_path + "_" + time.strftime('%m-%d_%H.%M', time.localtime())

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if args.architecture == "transformers":
        train_data, dev_data, test_data = build_dataset(config, n_samples=n_samples, sep=sep, multi_label=multi_label)
    else:
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word, n_samples=n_samples, lng=args.lng,
                                                               sep=sep, multi_label=multi_label)
        config.vocab = vocab
        config.n_vocab = len(vocab)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config)
    if args.architecture == 'DNN':
        init_network(model)
    print(model.parameters)
    print("Training...")
    start_time = time.time()
    train(config, model, train_iter, dev_iter, test_iter, loss_fn=loss_fn, multi_label=multi_label,
          cls_threshold=cls_threshold, eval_activate=eval_activate)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
