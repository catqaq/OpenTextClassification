<div style="font-size: 1.5rem;">
  <a href="./README.md">中文</a> |
  <a href="./docs/readme_en.md">English</a>
</div>
</br>

<h1 align="center">OpenTextClassification</h1>
<div align="center">
  <a href="https://github.com/catqaq/OpenTextClassification">
    <img src="https://pic4.zhimg.com/80/v2-f63d74cf9859eea57b0a78c9da00c9f3_720w.webp" alt="Logo" height="210">
  </a>

  <p align="center">
    <h3>Open text classification for you, Start your NLP journey</h3>
      <a href="https://github.com/catqaq/OpenTextClassification/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/catqaq/OpenTextClassification" />
      </a>
      <a href="https://github.com/kaixindelele/ChatPaper/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/catqaq/OpenTextClassification?color=0088ff" />
      </a>
      <a href="https://github.com/catqaq/OpenTextClassification/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/catqaq/OpenTextClassification?color=0088ff" />
      <a href="https://github.com/catqaq/OpenTextClassification/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/catqaq/OpenTextClassification?color=ccf" />
      </a>
      <br/>
      <em>开源实现 / 简单 / 全面 / 实践 </em>
      <br/>
      <a href="https://zhuanlan.zhihu.com/p/596112080/"><strong>文章解读</strong></a>
        ·
      <a href="https://zhuanlan.zhihu.com/p/596112080"><strong>视频解读</strong></a>
    </p>



 </p>
</div>

> **功能免费，代码开源，大家放心使用，欢迎贡献！**


- [💥最新讯息](#最新讯息)
- [💫开发计划](#开发计划)
- [⛏️使用步骤](#使用步骤)
- [📄运行示例](#运行示例)
- [📄结果展示](#结果展示)
- [🛠️常见报错](#常见报错)
- [💐参考资料&致谢](#参考资料&致谢)
- [🌟赞助我们](#赞助我们)
- [🌈Starchart](#Starchart)
- [🏆Contributors](#Contributors)




## 最新讯息

- 2023/03/23：OpenTextClassification V0.0.1版正式开源，版本特性：
  - 支持中英双语的文本分类
  - 支持多种文本分类模型：传统机器学习浅层模型、深度学习模型和transformers类模型
  - 支持多标签文本分类
  - 支持多种embedding方式：inner/outer/random

## 开发计划

OpenTextClassification项目为OpenNLP计划的第一个正式的开源项目，旨在Open NLP for everyone！在以ChatGPT/GPT4为代表的LLMs时代，在被OpenAI卷死之前，做一点有意义的事情！未来有一天，等到GPT-X发布的时候，或许有人会说NLP不存在了，但是我们想证明有人曾来过！



## 使用步骤

1.克隆本项目

`git clone https://github.com/catqaq/OpenTextClassification.git`

2.数据集下载和预处理

有空我再来补一个自动化脚本，暂时请自行处理或者参考preprocessing.py。

最好将数据统一放到data目录下，比如data/dbpedia，然后分3个子目录，input存放原始数据集（你下载的数据集），data存放预处理后的格式化的数据集（text-label格式），saved_dict存放训练结果（模型和日志等）。

3.运行示例

经过测试的开发环境如下，仅供参考，差不多的环境应该都可以运行。

- python：3.6/3.7
- torch：1.6.0
- transformers：4.18.0
- torchtext：0.7.0
- scikit-learn： 0.24.2
- tensorboardX：2.6
- nltk：3.6.7
- numpy：1.18.5
- pandas：1.1.5



根据自己的需要选择模块运行，详见下一节。

` python run.py`

## 运行示例

1.运行DNN/transformers类模型做文本分类

` python run.py`

2.运行传统浅层机器学习模型做文本分类

`python run_shallow.py`

3.运行DNN/transformers类模型做多标签文本分类

`python run_multi_label.py`



下表是直接运行demo的参考结果：

运行环境：python3.6 + T4

| demo               | 数据集      | 示例模型 | Acc    | 耗时      | 备注               |
| ------------------ | ----------- | -------- | ------ | --------- | ------------------ |
| run.py             | THUCNews/cn | TextCNN  | 89.94% | ~2mins    |                    |
| run_multi_label.py | rcv1/en     | bert     | 61.04% | ~40mins   | 其他指标见运行结果 |
| run_shallow.py     | THUCNews/cn | NB       | 89.44% | 105.34 ms |                    |

## 结果展示：待完善

笔者提供了从浅到深再到多标签的详细实验结果，可供大家参考。但受限于时间和算力，很多实验可能未达到最优，望知悉！因此，非常欢迎大家积极贡献，补充相关实验、代码和新的模型等等，一起建设OpenTextClassification。

暂时只提供部分汇总的结果，详细的实验结果及参数等我有空再补，比较多，需要一些时间整理。

| Data     | Model                    | tokenizer | 最小词长 | Min_df | ngram | binary | Use_idf | Test acc | 备注                                                         |
| -------- | ------------------------ | --------- | -------- | ------ | ----- | ------ | ------- | -------- | ------------------------------------------------------------ |
| THUCNews | LR                       | lcut      | 1        | 2      | (1,1) | False  | True    | 90.61%   | C=1.0, max_iter=1000  词表61549；  train score:  94.22%  valid score:  89.84%  test score: 90.61%  training time:  175070.97 ms |
|          | MultinomialNB(alpha=0.3) | lcut      | 1        | 2      | (1,1) | False  | True    | 89.86%   | 词表61549；  training time: 94.18ms                          |
|          | ComplementNB(alpha=0.8)  | lcut      | 1        | 2      | (1,1) | False  | True    | 89.88%   | 词表61549；  training time: 98.31ms                          |
|          | SVC(C=1.0)               | lcut      | 1        | 2      | (1,1) | False  | True    | 81.49%   | 词表61549；  维度200  training time:  7351155.59 ms  train score:  85.95%  valid score:  80.07%  test score: 81.49% |
|          | DT                       | lcut      | 1        | 2      | (1,1) | False  | True    | 71.19%   | max_depth=None     training time:  149216.53 ms  train score:  99.97%  valid score:  70.57%  test score: 71.19% |
|          | xgboost                  | lcut      | 1        | 2      | (1,1) | False  | True    | 90.08%   | XGBClassifier(n_estimators=2000,eta=0.3,gamma=0.1,max_depth=6,subsample=1,colsample_bytree=0.8,  nthread=10)  training time:  1551260.28 ms  train score:  99.00%  valid score:  89.34%  test score: 90.08% |
|          | KNN                      | lcut      | 1        | 2      | (1,1) | False  | True    | 83.34%   | k=5  training time:  22.14 ms  train score:  89.57%  valid score:  82.69%  test score: 83.34% |
|          |                          |           |          |        |       |        |         |          |                                                              |

 





## 常见报错



## 参考资料&致谢

A Survey on Text Classification: From Shallow to Deep Learning：https://arxiv.org/pdf/2008.00364.pdf?utm_source=summari

Deep Learning--based Text Classification: A Comprehensive Review：https://arxiv.org/pdf/2004.03705.pdf

https://github.com/649453932/Chinese-Text-Classification-Pytorch

https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

https://github.com/facebookresearch/fastText

https://github.com/brightmart/text_classification

https://github.com/kk7nc/Text_Classification

https://github.com/Tencent/NeuralNLP-NeuralClassifier

https://github.com/vandit15/Class-balanced-loss-pytorch

https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics



## 赞助我们

我们是谁？

我们是羡鱼智能【xianyu.ai】，主要成员是一群来自老和山下、西湖边上的咸鱼们，塘主叫作羡鱼，想在LLMs做点有意义的事！我们的口号是：做OpenNLP和OpenX！希望在OpenAI卷死我们之前退出江湖！

OpenTextClassification项目为羡鱼智能【xianyu.ai】发起的OpenNLP计划的第一个正式的开源项目，旨在Open NLP for everyone！在以ChatGPT/GPT4为代表的LLMs时代，在被OpenAI卷死之前，做一点有意义的事情！未来有一天，等到GPT-X发布的时候，或许有人会说NLP不存在了，但是我们想证明有人曾来过！

本项目第一版由本羡鱼利用业务时间（熬夜）独立完成，受限于精力和算力，拖延至今，好在顺利完成了。如果大家觉得本项目对你的NLP学习/研究/工作有所帮助的话，求一个免费的star! 富哥富姐们可以考虑赞助一下！尤其是算力，租卡的费用已经让我颇为肉疼了。

<img src="https://xianyunlp.oss-cn-hangzhou.aliyuncs.com/uPic/image-20230324010955205.png" alt="image-20230324010955205" style="zoom: 25%;" />

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=catqaq/OpenTextClassification&type=Date)](https://star-history.com/#kaixindelele/ChatPaper&Date)

## Contributors

<a href="https://github.com/catqaq/OpenTextClassification/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=catqaq/OpenTextClassification" />
</a>