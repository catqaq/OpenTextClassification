# Text Classification
- Dataset: yahoo_answers_csv ([yahoo](https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz))
- Model: CNN/LSTM/GRU.
- Pytorch: 1.0.1
- Python: 3.6
- torchtext: 0.3.1.
- Support pretrained word embedding.([word2vec](https://github.com/mmihaltz/word2vec-GoogleNews-vectors))

## torchtext
- This package can provide an elegant way to build vocabulary([torchtext](https://torchtext.readthedocs.io/en/latest/index.html#)). 
```
TEXT.build_vocab(dataset, vectors)
```

## Training

- The following command starts training.

```
python main.py
```
## Refs
- Kim, Y. J. E. A. (2014). "Convolutional Neural Networks for Sentence Classification." 
- ([cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch))
- ([Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
