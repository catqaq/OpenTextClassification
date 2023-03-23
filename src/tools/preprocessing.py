import pandas as pd
from nltk import word_tokenize
import os
import jsonlines
import re

# concatenate all necessary text field except the label, id field
from sklearn.model_selection import train_test_split


def preprocess(data_path, header=None, skiprows=0, sep=",", text_fields=[0], text_fields_weight=None, label_field=1,
               unused_fields=[], save_path=None, save_sep=None):
    df = pd.read_csv(data_path, header=header, skiprows=skiprows, sep=sep)
    df.fillna('', inplace=True)  # text字段缺失值默认填充空字符
    if not text_fields_weight:
        text_fields_weight = [[1] * len(text_fields) for i in range(len(df))]
    df['text'] = [" ".join(t * w) for t, w in zip(df[text_fields].values, text_fields_weight)]
    if label_field is not None:
        df['label'] = df[label_field]
        if df['label'].min() > 0:
            df['label'] = df['label'] - df['label'].min()
    df.drop(columns=unused_fields, inplace=True)
    if save_path is not None:
        if save_sep is None:
            save_sep = sep
        if label_field is not None:
            df[["text", "label"]].to_csv(save_path, sep=save_sep, header=None, index=False)
        else:
            df[["text"]].to_csv(save_path, sep=save_sep, header=None, index=False)
    return df


def split_train_dev(train_path, dev_ratio=0.1, random_state=42):
    with open(train_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    train, dev = train_test_split(lines, test_size=dev_ratio, random_state=random_state)
    with open(train_path, "w", encoding="utf8") as f:
        f.writelines(train)
    with open(train_path.replace("train.", "dev."), "w", encoding="utf8") as f:
        f.writelines(dev)


def preprocess_rcv1_json(in_path_list, out_path_list=None, n_samples=None, save_classes=False):
    def read_json(in_path):
        with jsonlines.open(in_path) as f:
            lines = [
                (' '.join(line['doc_token']), ' '.join(line['doc_topic']), ' '.join(line['doc_keyword']),
                 line['doc_label']) for line in f]
            if n_samples:
                lines = lines[:n_samples]
        return lines

    train = read_json(in_path_list[0])
    dev = read_json(in_path_list[1])
    test = read_json(in_path_list[2])
    lines = train + dev + test
    label_map = {}
    for line in lines:
        for label in line[3]:
            label_map[label] = label_map.get(label, 0) + 1
    print("label_map sort by freq...")
    labels = sorted([item for item in label_map.items()], key=lambda x: x[1], reverse=True)
    label_map = {item[0]: i for i, item in enumerate(labels)}
    print(label_map)
    train = [' '.join(line[:-1]) + "\t" + ','.join([str(label_map.get(label)) for label in line[-1]]) + "\n" for
             line in train]
    dev = [' '.join(line[:-1]) + "\t" + ','.join([str(label_map.get(label)) for label in line[-1]]) + "\n" for
           line in dev]
    test = [' '.join(line[:-1]) + "\t" + ','.join([str(label_map.get(label)) for label in line[-1]]) + "\n" for
            line in test]
    classes = [item[0] + "\n" for item in labels]
    total = [train, dev, test]
    for i in range(len(out_path_list)):
        with open(out_path_list[i], 'w', encoding='utf8') as f:
            f.writelines(total[i])
    if save_classes:
        classes_path = re.sub('[^/]+(?!.*/)', 'class.txt', out_path_list[0])
        with open(classes_path, 'w', encoding='utf8') as f:
            f.writelines(classes)


if __name__ == "__main__":
    # data_root = "../../data/"
    # # dbpedia
    # dataset = "dbpedia"
    # data_path = data_root + dataset + "/input/train.csv"
    # save_path = data_path.replace("/input/", "/data/").replace(".csv", ".txt")
    # preprocess(data_path, header=None, sep=",", text_fields=[1, 2], text_fields_weight=None, label_field=0,
    #            save_path=save_path)
    # split_train_dev(save_path, 0.1)
    # data_path = data_root + dataset + "/input/test.csv"
    # save_path = data_path.replace("/input/", "/data/").replace(".csv", ".txt")
    # preprocess(data_path, header=None, sep=",", text_fields=[1, 2], text_fields_weight=None, label_field=0,
    #            save_path=save_path)

    # # nlp-with-disaster-tweets
    # dataset = "nlp-with-disaster-tweets"
    # data_path = data_root + dataset + "/input/train.csv"
    # save_path = data_path.replace("/input/", "/data/").replace(".csv", ".txt")
    # preprocess(data_path, header=None, skiprows=1, sep=",", text_fields=[1, 2, 3], text_fields_weight=None,
    #            label_field=4, save_path=save_path)
    # data_path = data_root + dataset + "/input/test.csv"
    # save_path = data_path.replace("/input/", "/data/").replace(".csv", ".txt")
    # preprocess(data_path, header=None, skiprows=1, sep=",", text_fields=[1, 2, 3], text_fields_weight=None,
    #            label_field=None, save_path=save_path)

    in_path_list = ['../../data/rcv1/data/rcv1_train.json', '../../data/rcv1/data/rcv1_dev.json',
                    '../../data/rcv1/data/rcv1_test.json']
    out_path_list = ['../../data/rcv1/data/train.txt', '../../data/rcv1/data/dev.txt',
                     '../../data/rcv1/data/test.txt']

    preprocess_rcv1_json(in_path_list, out_path_list, n_samples=None, save_classes=True)
