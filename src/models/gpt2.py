# coding: UTF-8
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset, pretrained_name_or_path=None):
        self.model_name = 'gpt2'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.multi_label = False
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.encoder_path = './gpt2_pretrain' if not pretrained_name_or_path else pretrained_name_or_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.encoder_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = GPT2Model.from_pretrained(config.encoder_path)
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # 模型输出：
        lengths = torch.sum(mask, dim=1) - 1  # 减去1是因为序列索引是从0开始的
        pooled = self.encoder(context, attention_mask=mask, return_dict=True).last_hidden_state
        last_indices = lengths.unsqueeze(1).unsqueeze(2).expand(-1, -1, pooled.size(-1))  # 扩展维度以与 pooled 匹配
        pooled = torch.gather(pooled, dim=1, index=last_indices).squeeze(1)
        out = self.fc(pooled)
        return out
