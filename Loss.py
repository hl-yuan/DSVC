import math

import torch
import torch.nn.functional as F
from utils import get_Similarity
import torch.nn as nn



class Loss_Reconstruction(nn.Module):

    def __init__(self):
        super(Loss_Reconstruction, self).__init__()

    def forward(self, xs, xr):
        RF = torch.nn.MSELoss(reduction='mean').cuda()

        loss = RF(xs, xr)
        return loss



class Loss_Konw_Guide(nn.Module):

    def __init__(self):
        super(Loss_Konw_Guide, self).__init__()

    def forward(self, proto, feature):
        feature = F.normalize(feature, dim=1)

        proto = F.normalize(proto, dim=1)
        feature[torch.isinf(feature)] = 0
        feature[torch.isnan(feature)] = 0
        proto[torch.isinf(proto)] = 0
        proto[torch.isnan(proto)] = 0
        tau = 1
        sim_matrix = torch.matmul(proto, feature.T) / tau
        log_softmax_sim_matrix = F.log_softmax(sim_matrix, dim=1)
        loss = -log_softmax_sim_matrix.mean()

        return loss


class Cross_contrast(nn.Module):

    def __init__(self):
        super(Cross_contrast, self).__init__()
    def forward(self, h0, h1):

        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        cos12 = get_Similarity(h0, h1)
        cos11 = get_Similarity(h0, h0)
        cos22 = get_Similarity(h1, h1)
        cos21 = get_Similarity(h1, h0)
        sim12 = (cos12).exp()
        sim11 = (cos11).exp()
        sim21 = (cos21).exp()
        sim22 = (cos22).exp()
        pos1 = sim12.diag()
        pos2 = sim21.diag()
        p1 = pos1 / (sim12 + sim11).sum(1)
        p2 = pos2 / (sim21 + sim22).sum(1)
        loss1 = (p1 * (torch.log(p1 / p2))).sum()
        loss2 = (p2 * (torch.log(p2 / p1))).sum()

        loss = (loss1+loss2)/2
        return loss




