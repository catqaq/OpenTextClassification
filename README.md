# Text Classification
- Dataset: [yahoo/dbpedia...](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
- Model: CNN/(Bi)LSTM/(Bi)GRU.
- Pytorch: 1.0.1
- Python: 3.6
- torchtext: 0.3.1.
- Support pretrained word embedding.([word2vec](https://github.com/mmihaltz/word2vec-GoogleNews-vectors),[glove](https://nlp.stanford.edu/projects/glove/))

## torchtext
- This package can provide an elegant way to build vocabulary([torchtext](https://torchtext.readthedocs.io/en/latest/index.html#)). 
```
TEXT.build_vocab(dataset, vectors)
```
## Training

```
python preprocess.py #preprocessing
```
```
python main.py #training
```
## Refs
- Kim, Y. J. E. A. (2014). "Convolutional Neural Networks for Sentence Classification." 
- [cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Glove](https://nlp.stanford.edu/projects/glove/)
- [你在训练RNN的时候有哪些特殊的trick？](https://www.zhihu.com/question/57828011)
- [CNN](https://blog.csdn.net/v_JULY_v/article/details/51812459)
- [rnn.py](pytorch/pytorch/blob/master/torch/nn/modules/rnn.py)
- [rnn initialization](https://discuss.pytorch.org/t/lstm-gru-gate-weights/2807)
- [Explaining and illustrating orthogonal initialization for recurrent neural networks](https://smerity.com/articles/2016/orthogonal_init.html)
