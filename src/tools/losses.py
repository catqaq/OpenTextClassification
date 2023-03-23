import torch


def cross_entropy_multi_label(logits, y):
    """similar to softmax_cross_entropy_with_logits_v2 in tensorflow 1.x"""
    s = torch.exp(logits)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)


def multi_label_circle_loss(logits, y, reduction='mean', inf=1e12, threshold=0):
    # ref:https://kexue.fm/archives/7359
    # pytorch实现：https://github.com/yongzhuo/pytorch-loss or https://bbs.hankcs.com/t/topic/4022
    logits = (1 - 2 * y) * logits  # <B, C>
    logits_neg = logits - y * inf  # <B, C>
    logits_pos = logits - (1 - y) * inf  # <B, C>
    zeros = torch.zeros_like(logits[..., :1])  # <B, 1>
    zeros = zeros + threshold
    logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <B, C+1>
    logits_pos = torch.cat([logits_pos, -zeros], dim=-1)  # <B, C+1>
    neg_loss = torch.logsumexp(logits_neg, dim=-1)  # <B, >
    pos_loss = torch.logsumexp(logits_pos, dim=-1)  # <B, >
    loss = neg_loss + pos_loss
    if "mean" == reduction:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss


if __name__ == "__main__":
    y_true = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]])
    y_pred = torch.tensor([[0.2, 0.5, 0.1, 0], [0.1, 0.5, 0, 0.8]])
    loss = cross_entropy_multi_label(y_pred, y_true)
    print(loss)  # 2.3928
