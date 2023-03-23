# coding: UTF-8
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from sklearn.metrics import multilabel_confusion_matrix

from tools.utils import get_time_dif
from tensorboardX import SummaryWriter
from transformers import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, loss_fn=None, multi_label=False, cls_threshold=0,
          eval_activate=None):
    start_time = time.time()
    model.train()
    model.to(config.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Migrating from pytorch-pretrained-bert: To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    num_training_steps = config.num_epochs * len(train_iter)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_training_steps,
                                                num_training_steps=num_training_steps)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    loss_fn = loss_fn if loss_fn else F.cross_entropy
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            if loss_fn == F.binary_cross_entropy_with_logits:
                labels = labels.float()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                if multi_label:
                    outputs = get_activations(outputs, eval_activate)
                    predic = torch.where(outputs > cls_threshold, torch.ones_like(outputs),
                                         torch.zeros_like(outputs)).cpu()
                else:
                    predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter, loss_fn=loss_fn, multi_label=multi_label,
                                             cls_threshold=cls_threshold, eval_activate=eval_activate)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter, loss_fn=loss_fn, multi_label=multi_label, cls_threshold=cls_threshold,
         eval_activate=eval_activate)


def test(config, model, test_iter, loss_fn=None, multi_label=False, cls_threshold=0, eval_activate=None):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True, loss_fn=loss_fn,
                                                                multi_label=multi_label, cls_threshold=cls_threshold,
                                                                eval_activate=eval_activate)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False, loss_fn=None, multi_label=False, cls_threshold=0,
             eval_activate=None):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    loss_fn = loss_fn if loss_fn else F.cross_entropy
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            if loss_fn == F.binary_cross_entropy_with_logits:
                labels = labels.float()
            loss = loss_fn(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            if multi_label:
                outputs = get_activations(outputs, eval_activate)
                predic = torch.where(outputs > cls_threshold, torch.ones_like(outputs),
                                     torch.zeros_like(outputs)).cpu().numpy()
            else:
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all.append(labels)
            predict_all.append(predic)

    labels_all, predict_all = np.concatenate(labels_all), np.concatenate(predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, labels=list(range(config.num_classes)),
                                               target_names=config.class_list, digits=4)
        if multi_label:
            confusion = multilabel_confusion_matrix(labels_all, predict_all)
        else:
            confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def get_activations(inputs, activate_func=None):
    if activate_func == 'sigmoid':
        inputs = torch.sigmoid(inputs)
    elif activate_func == 'softmax':
        inputs = torch.softmax(inputs)
    elif activate_func == 'tanh':
        inputs = torch.tanh(inputs)
    return inputs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
