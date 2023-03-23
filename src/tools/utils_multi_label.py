import torch


def get_multi_hot_label(doc_labels, label_size, dtype=torch.long):
    """For multi-label classification
    Generate multi-hot for input labels
    e.g. input: [[0,1], [2]]
         output: [[1,1,0], [0,0,1]]
    """
    batch_size = len(doc_labels)
    max_label_num = max([len(x) for x in doc_labels])
    doc_labels_extend = \
        [[doc_labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]
    for i in range(0, batch_size):
        doc_labels_extend[i][0: len(doc_labels[i])] = doc_labels[i]
    y = torch.Tensor(doc_labels_extend).long()
    y_onehot = torch.zeros(batch_size, label_size, dtype=dtype).scatter_(1, y, 1)
    return y_onehot


if __name__ == "__main__":
    pass
