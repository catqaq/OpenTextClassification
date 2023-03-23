import time
from datetime import timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from nltk import word_tokenize
from nltk.corpus import stopwords
from jieba import lcut
import string

# import nltk

# nltk.download('stopwords')

from tools.utils_shallow import build_dataset

# 数据
with_test_label = True  # 某些数据集，如比赛数据，测试集标签可能无法获取
lng = 'cn'
data_path = "./data/"
dataset = 'THUCNews'  # 数据集
sep = "\t"
# lng = 'en'
# dataset = 'dbpedia'  # 数据集
# sep = ","
# lng = 'cn'
# dataset = 'nlp-with-disaster-tweets'  # 数据集
# sep = "\t"
dataset = data_path + dataset
n_samples = None

# 超参数
# 模型相关参数
model = "NB"
# 特征相关参数
cut = True if lng == 'cn' else False
# cut = False
tokenizer = lcut if cut else None
# tokenizer = None
# The default regexp select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always
# treated as a token separator).
token_pattern = r"(?u)\b\w+\b"  # >=1个字符，适合中文
# token_pattern = r"(?u)\b\w\w+\b"  # >=2个字符，适合英文
strip_accents = 'unicode'  # 默认为None, 可选ascii,unicode
# strip_accents = None
min_df = 2
ngram_range = (1, 1)  # 中文如果做了分词，这里取(1,1)即只有unigram，否则会出现用空格进行连接的中文词语
max_features = None
binary = False  # 将tf设置成0/1
use_idf = True
sublinear_tf = False

en_punctuation = list(string.punctuation)
# 中文标点来自于zhon.hanzi.punctuation
cn_punctuation = list('＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。')
punctuations = en_punctuation + cn_punctuation
punctuations = sorted(set(punctuations), key=punctuations.index)
print(punctuations)
stop_words = None
# stop_words = stopwords.words('english')
stop_words = punctuations if not stop_words else stop_words + punctuations
# Tokenizing the stop words generated tokens [' ', '...', '⦅', '⦆'] not in stop_words.
stop_words.extend([' ', '...', '⦅', '⦆'])
stop_words = None

if __name__ == "__main__":
    # 1.load data
    train, valid, test = build_dataset(dataset, sep=sep, tokenizer=None, n_samples=n_samples)
    xtrain, ytrain = train[0].values, train[1].values
    xvalid, yvalid = valid[0].values, valid[1].values
    xtest, ytest = test[0].values, test[1].values if with_test_label else None

    # 2.get features
    vectorizer = TfidfVectorizer(min_df=min_df, max_features=None, strip_accents=strip_accents, analyzer='word',
                                 tokenizer=tokenizer, token_pattern=token_pattern, ngram_range=ngram_range,
                                 binary=binary, use_idf=use_idf, smooth_idf=True, sublinear_tf=sublinear_tf,
                                 stop_words=stop_words)

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    vectorizer.fit(list(xtrain) + list(xvalid))  # 这里要合并，词表和idf应该基于完整的训练集构建，注意这里的valid是从原train中分出来的
    # Transform documents to document-term matrix.
    xtrain_dtm = vectorizer.transform(xtrain)
    xvalid_dtm = vectorizer.transform(xvalid)
    xtest_dtm = vectorizer.transform(xtest)

    print(len(vectorizer.vocabulary_))  # 词汇表大小
    print(sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:5])  # 词汇表：top5的高频词
    print(vectorizer.get_feature_names()[:5])
    df = pd.DataFrame(xtrain_dtm.toarray(), columns=vectorizer.get_feature_names())
    print(df)

    # 3.classification
    if model == 'LR':
        clf = LogisticRegression(C=1.0, max_iter=1000)
    elif model == 'NB':
        # 多项式分布NB和CNB是文本分类中常用的NB
        # CNB是标准多项式朴素贝叶斯(MNB)算法的一种改进，特别适用于不平衡数据集，在文本分类任务上通常比MNB表现得更好(通常有相当大的优势)
        clf = MultinomialNB(alpha=1.0)
        # clf = ComplementNB(alpha=1.0)  # 平滑因子的影响很大
    elif model == 'SVM':
        # You can create the pipeline: svd, scl, clf...
        # 降维算法
        # Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
        dec = decomposition.TruncatedSVD(n_components=100)
        dec.fit(xtrain_dtm)
        xtrain_dtm = dec.transform(xtrain_dtm)
        xvalid_dtm = dec.transform(xvalid_dtm)
        xtest_dtm = dec.transform(xtest_dtm)
        # PCA：无法使用
        # dec = decomposition.PCA(n_components=120) #PCA不支持稀疏数据
        # dec = decomposition.SparsePCA(n_components=120)
        # dec.fit(xtrain_dtm.toarray())
        # xtrain_dtm = dec.transform(xtrain_dtm.toarray())
        # xvalid_dtm = dec.transform(xvalid_dtm.toarray())
        # xtest_dtm = dec.transform(xtest_dtm.toarray())

        # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
        scl = preprocessing.StandardScaler()  # 先做标准化
        scl.fit(xtrain_dtm)
        xtrain_dtm = scl.transform(xtrain_dtm)
        xvalid_dtm = scl.transform(xvalid_dtm)
        xtest_dtm = scl.transform(xtest_dtm)
        # SVM
        clf = SVC(C=1.0, max_iter=1000)
        # Kernel Approximation + LinearSVC/SGDClassifier
        ka = RBFSampler(gamma=0.001, n_components=100, random_state=1)
        # ka = Nystroem(gamma=0.2, n_components=100, random_state=1)
        # ValueError: Negative values in data passed to X in AdditiveChi2Sampler.fit
        # ka = AdditiveChi2Sampler(sample_steps=2)
        ka.fit(xtrain_dtm)
        xtrain_dtm = ka.transform(xtrain_dtm)
        xvalid_dtm = ka.transform(xvalid_dtm)
        xtest_dtm = ka.transform(xtest_dtm)
        # clf = LinearSVC(C=1.0, max_iter=1000)
        clf = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=100, n_jobs=4)
    elif model == 'DT':
        clf = DecisionTreeClassifier(max_depth=500, min_samples_leaf=5)
    elif model == 'GBDT':
        xtrain_dtm = xtrain_dtm.tocsc()
        xvalid_dtm = xvalid_dtm.tocsc()
        xtest_dtm = xtest_dtm.tocsc()
        # 默认参数，除了线程数以外
        # clf = xgb.XGBClassifier(n_estimators=100, eta=0.3, gamma=0, max_depth=6, subsample=1, colsample_bytree=1,
        #                         nthread=10, reg_alpha=0, reg_lambda=1)
        # xgboost调参很有讲究
        clf = xgb.XGBClassifier(n_estimators=100, eta=0.3, gamma=0.1, max_depth=6, subsample=1, colsample_bytree=0.8,
                                nthread=10, reg_alpha=0, reg_lambda=1)
    elif model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    else:
        clf = LogisticRegression(C=1.0, max_iter=1000)

    # train
    print('start training......')
    start = time.time()
    clf.fit(xtrain_dtm, ytrain)
    end = time.time()
    print(f'training time: {(end - start) * 1000:.2f} ms')
    # predict
    # y_pred = clf.predict(xtrain_dtm)
    score = clf.score(xtrain_dtm, ytrain)
    print(f'train score: {score:.2%}')
    # y_pred = clf.predict(xvalid_dtm)
    score = clf.score(xvalid_dtm, yvalid)
    print(f'valid score: {score:.2%}')
    # y_pred = clf.predict(xtest_dtm)
    score = clf.score(xtest_dtm, ytest)
    print(f'test score: {score:.2%}')
