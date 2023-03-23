import pandas as pd


def build_dataset(dataset, sep="\t", tokenizer=None, n_samples=None):
    def load_dataset(path):
        df = pd.read_csv(path, sep=sep, header=None, nrows=n_samples)
        df[0] = df[0].apply(lambda x: " ".join(tokenizer(x)) if tokenizer else x)
        return df

    train_path = dataset + '/data/train.txt'
    dev_path = dataset + '/data/dev.txt'
    test_path = dataset + '/data/test.txt'
    train = load_dataset(train_path)
    dev = load_dataset(dev_path)
    test = load_dataset(test_path)
    return train, dev, test


if __name__ == "__main__":
    pass
