import numpy as np
from sklearn.cluster import KMeans
import data_loader
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import time
import itertools
from Knowledge_Generation import dual_attention
from Nmetrics import evaluate
import torch
from network import Autoencoder
import pandas as pd
from Streaming_train import train_view_0, train_view

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class StreamingVC():
    def __init__(self, config):

        self.feature_dim = config.feature_dim
        print(config.datasets)
        self.X, self.Y = data_loader.load_data2(config.datasets)
        self.datasets = config.datasets
        self.cluster, self.v = config.Cluster, config.view_num
        # self.data = data
        self.n_view = config.view_num
        self.Knowledge_size = config.Knowledge_size

        self.word_dim = 256
        self.dropout_prob = 0.5
        self.input_dim = [self.X[v].shape[1] for v in range(self.n_view)]
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.view_id = config.view_num
        self.epochs = config.epochs
        self.cuda_list = config.cuda_list
        self.config = config

        self.runing_time = config.running_time


    def train(self, path, filename, loss_para):

        print("****************  begin training ****************")
        start = time.time()
        Buffer = []
        for _ in range(2):
            Buffer.append(None)
        Stre_fea = []

        for v in range(self.n_view):

            self.input_X_dim = self.input_dim[v]
            print("**************** dataset:{}, view:{}, input_dim:{}, Cluster_number:{}  ****************".format(self.datasets[1], v, self.input_X_dim ,self.cluster))

            Dual_attention = dual_attention(dim=self.feature_dim, num=self.Knowledge_size)
            decoder_model = Autoencoder(self.input_X_dim, self.feature_dim, activation='relu', batchnorm=True)
            optimizer_Streaming = torch.optim.Adam(itertools.chain(decoder_model.parameters(), Dual_attention.parameters()), lr=self.lr)
            decoder_model = decoder_model.cuda()
            Dual_attention = Dual_attention.cuda()

            if v == 0:
                fea_end, Buffer_0 = train_view_0(decoder_model, Dual_attention, optimizer_Streaming, self.X[v], self.Y, v, path, filename, self.config, self.cluster, Buffer, loss_para)
                Buffer = Buffer_0

            else:
                fea_end, Buffer_v = train_view(decoder_model, Dual_attention, optimizer_Streaming, self.X[v], self.Y, v, path, filename, self.config, self.cluster, Buffer, loss_para)
                Buffer = Buffer_v
            Stre_fea.append(fea_end.detach().cpu().numpy())


            for j in range(v):
                Stre_fea[j] = Stre_fea[j]
            Labels = self.Y[0]
            estimator = KMeans(n_clusters=self.cluster)

            Stre_cluster = Stre_fea[0]
            for i in range(1, len(Stre_fea)):
                Stre_cluster = np.concatenate((Stre_cluster, Stre_fea[i]), axis=1)

            estimator.fit(Stre_cluster)
            pred_final = estimator.labels_

            acc, nmi, purity, fscore, precision, recall, ari = evaluate(Labels, pred_final)
            print("result", "acc=", acc * 100, "nmi=", nmi * 100, "fscore=", fscore * 100, "ari=", ari * 100, "recall=", recall * 100, "precision=", precision * 100, "purity=", purity*100)

        end = time.time()
        runing_time = end - start
        print("--------over training:{}--------".format(runing_time))
        return acc * 100, nmi * 100, fscore * 100, ari * 100, recall * 100, precision * 100, purity*100



