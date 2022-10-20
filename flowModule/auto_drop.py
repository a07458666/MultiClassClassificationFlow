
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def auto_drop(log_p2, batch_size, correct):
    # auto decide drop rate
    kmeans = KMeans(n_clusters=2).fit(-log_p2)
    centers = kmeans.cluster_centers_
    if centers[0][0] < centers[1][0]:
        drop_rate = kmeans.labels_.sum() / batch_size
    else:
        drop_rate = 1. - (kmeans.labels_.sum() / batch_size)

    # get drop mask
    drop_n = int(drop_rate * batch_size)
    if drop_n > 0 and drop_n < batch_size:
        loss_top = torch.topk(-log_p2, drop_n, dim=0)
        drop_mask = torch.nn.functional.one_hot(loss_top.indices.squeeze(), batch_size).view(-1, batch_size).sum(dim=0).unsqueeze(1)
    else:
        drop_mask = torch.ones(size=log_p2.size())

    # statistic drop data
    tn, fp, fn, tp = confusion_matrix((1-correct).cpu().detach().squeeze(), drop_mask.cpu().detach().squeeze()).ravel()
    drop_precision = tp / (tp + fp + 1e-10)
    drop_recall    = tp / (tp + fn + 1e-10)
    drop_acc       = (tp + tn) / (tp + fp + fn + tn)
    # drop sample has only 10% loss
    drop_mask = drop_mask * 0.9

    return drop_mask, drop_precision, drop_recall, drop_acc, drop_rate