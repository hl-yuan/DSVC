import torch
import torch.nn.functional as F
# from kmeans_gpu import kmeans
import math
import numpy as np
import time
import sys
import os
import logging
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NormalizeFeaTorch(features):
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    normalized_feas = torch.mm(r_mat_inv, features)
    return normalized_feas



def get_Similarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat



def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



def next_batch1(X1, X2, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]

    total = math.ceil(tot / batch_size)
    X2 = torch.tensor(X2)
    X2 = X2.t()
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)

        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]

        yield (batch_x1, X2, (i + 1))


def next_batch(X1, X2, Proto, Attention, batch_size, Knowledge_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    tot_p = Proto.shape[0]
    total = math.ceil(tot / batch_size)
    X2 = torch.tensor(X2)
    X2 = X2.t()
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)

        start_idx_p = i * Knowledge_size
        end_idx_p = (i + 1) * Knowledge_size
        end_idx_p = min(tot_p, end_idx_p)

        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_Atten = Attention[start_idx: end_idx, ...]
        batch_Proto = Proto[start_idx_p: end_idx_p, ...]

        yield (batch_x1, X2, batch_Proto, batch_Atten, (i + 1))

def cal_std(*arg):
    """Return the average and its std"""
    if len(arg) == 6:
        # logger.info('ACC:' + str(arg[0]))
        # logger.info('NMI:' + str(arg[1]))
        # logger.info('f_mea:' + str(arg[2]))
        # logger.info('ARI:' + str(arg[3]))
        # logger.info('recall:' + str(arg[4]))
        # logger.info('precision:' + str(arg[5]))
        output = """ 
                    ACC {:.2f} std {:.2f} 
                    NMI {:.2f} std {:.2f} 
                    f_mea {:.2f} std {:.2f} 
                    ARI {:.2f} std {:.2f} 
                    recall {:.2f} std {:.2f} 
                    precision {:.2f} std {:.2f}
                    purity {:.2f} std {:.2f}""".format(
                     np.mean(arg[0]), np.std(arg[0]),
                     np.mean(arg[1]), np.std(arg[1]),
                     np.mean(arg[2]), np.std(arg[2]),
                     np.mean(arg[3]), np.std(arg[3]),
                     np.mean(arg[4]), np.std(arg[4]),
                     np.mean(arg[5]), np.std(arg[5]),
                     np.mean(arg[6]), np.std(arg[6]))

    return np.mean(arg[0]), np.std(arg[0]), np.mean(arg[1]), np.std(arg[1]), np.mean(arg[2]), np.std(arg[2]), np.mean(arg[3]), np.std(arg[3]), np.mean(arg[4]), np.std(arg[4]), np.mean(arg[5]), np.std(arg[5]), np.mean(arg[6]), np.std(arg[6])

def get_logger(config):

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
        if not os.path.exists('./logs/' + str(config.datasets[1]) + '/'):
            os.mkdir('./logs/' + str(config.datasets[1]) + '/')
    path = os.path.join("./logs/" + str(config.datasets[1]) + '/')
    if not os.path.exists(path):
        os.makedirs(path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_time = current_time.replace(":", "_")
    filename = config.datasets[1] + "_time=" + current_time + ".csv"
    results_df = pd.DataFrame(columns=["View", "Accuracy", "NMI", "Fscore", "ARI", "Recall", "Precision", "Loss"])
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)