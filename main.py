import torch
import argparse
import data_loader
from Streaming_model import StreamingVC
import numpy as np
import random as rn
from torch.backends import cudnn
import logging
import time
import sys
import os
import collections
import pandas as pd
from utils import *

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def set_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False


def main(config):
    seed = [11, 21, 22, 123, 456]

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
    results_df = pd.DataFrame(
        columns=["View", "Accuracy", "NMI", "Fscore", "ARI", "Recall", "Precision", "Purity", "Loss"])
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    results_df.to_csv(path + filename, index=False)
    logging.info(config)
    loss_para = [0, 0]
    loss_para[0] = config.beta1
    loss_para[1] = config.beta2
    accumulated_metrics = collections.defaultdict(list)
    for i in range(len(seed)):
        logging.info("--------seed:{}--------".format(seed[i]))
        np.random.seed(seed[i])
        rn.seed(seed[i])
        os.environ['PYTHONHASHSEED'] = str(seed[i])
        torch.cuda.empty_cache()
        torch.manual_seed(seed[i])
        torch.cuda.manual_seed(seed[i])
        torch.cuda.manual_seed_all(seed[i])
        cudnn.enabled = False

        print("Information", config)
        print('seed: ' + str(seed[i]))

        DSVC = StreamingVC(config)
        acc, nmi, fscore, ari, recall, precision, purity = DSVC.train(path, filename, loss_para)
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['f_mea'].append(fscore)
        accumulated_metrics['ari'].append(ari)
        accumulated_metrics['recall'].append(recall)
        accumulated_metrics['precision'].append(precision)
        accumulated_metrics['purity'].append(purity)

    metrics_summary = cal_std(
        accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['f_mea'],
        accumulated_metrics['ari'], accumulated_metrics['recall'],
        accumulated_metrics['precision'], accumulated_metrics['purity']
    )

    mean_metrics = ["mean"] + [f"{x:.4f}" for x in metrics_summary[::2]]
    std_metrics = ["std"] + [f"{x:.4f}" for x in metrics_summary[1::2]]
    print(mean_metrics)
    print(std_metrics)
    pd.DataFrame([mean_metrics]).to_csv(path + filename, mode='a', header=False, index=False)
    pd.DataFrame([std_metrics]).to_csv(path + filename, mode='a', header=False, index=False)


i_d = {
    1: "ALOI_10",
    2: "HandWritten",
    3: "ALOI_100",
    4: "YoutubeFace_sel_fea",
    5: "stl10_fea",
    6: "LandUse_21",
    7: "Scene_15",
    8: "ALOI_1000",
}

if __name__ == '__main__':

    selected_datasets = [1, 2, 3, 4, 5, 6, 7, 8]
    for dataset_id in selected_datasets:
        dataset_name = i_d[dataset_id]
        print(f"Processing dataset: {dataset_name}")
        my_data_dic = data_loader.ALL_data
        data_para = my_data_dic[dataset_name]
        parser = argparse.ArgumentParser()
        parser.add_argument('--i_d', type=int, default=dataset_id)
        parser.add_argument('--datasets', default=data_para)
        parser.add_argument('--compute_all', type=bool, default=False)
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--running_time', type=bool, default=False)
        parser.add_argument('--cuda_list', type=list, default=[0])
        parser.add_argument('--lr', type=list, default=1e-4)
        parser.add_argument('--beta1', type=float, default=data_para['beta1'])
        parser.add_argument('--beta2', type=float, default=data_para['beta2'])
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--feature_dim', type=int, default=256)
        parser.add_argument('--Knowledge_size', type=int, default=data_para['Knowledge_size'])
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--view_num', type=int, default=data_para['V'])
        parser.add_argument('--Cluster', type=int, default=data_para['K'])
        parser.add_argument('--sample_interval', type=int, default=1)
        parser.add_argument('--epochs', type=int, default=200)
        config = parser.parse_args()
        main(config)