import torch.nn.functional as F
import torch.nn as nn

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

def js_distance(pred, target, num_class):
    JS_dist = Jensen_Shannon()
    dist = JS_dist(pred, F.one_hot(target, num_classes = num_class))
    return dist