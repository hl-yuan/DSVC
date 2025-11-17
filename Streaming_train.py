import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from torch import optim
import torch.nn.functional as F
import data_loader
from torch.autograd import Variable
import scipy.io as sio
import copy
import math
import time
import random as rn
from tqdm import tqdm
from Nmetrics import evaluate
import torch
from network import Autoencoder
import pandas as pd
import os
# from Knowledge_Generation import Knowledge_Memory
from Loss import Loss_Reconstruction, Loss_Konw_Guide, Cross_contrast
from Knowledge_Generation import dual_attention
from utils import next_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_view_0(decoder_model, Dual_attention, optimizer_Streaming, Xv, Y, v, path, filename, config, cluster, Buffer, loss_para):
    seed = 0
    np.random.seed(seed)
    rn.seed(seed)

    train_dataset = data_loader.TrainDataset_Com(Xv, Y)
    batch_sampler = data_loader.Data_Sampler(train_dataset, shuffle=True, batch_size=config.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    t_progress = tqdm(range(config.epochs), desc=f'Streaming_View{v}_training')

    for epoch in t_progress:
        index_swap = None
        tot_loss = 0.0
        Proto_total = []
        Atten_total = []
        Rec_loss = Loss_Reconstruction().cuda()
        Guide_loss = Loss_Konw_Guide().cuda()
        Cross_contrast_loss = Cross_contrast().cuda()


        for batch_idx, (xs, ys) in enumerate(train_loader):

            xs = np.squeeze(xs, axis=0)
            xs = torch.tensor(xs).cuda()

            decoder_model = decoder_model.cuda()
            Dual_attention = Dual_attention.cuda()
            xs = xs.squeeze(0)
            z, xr = decoder_model(xs)

            re_loss = Rec_loss(xs, xr)

            z = z.squeeze(0)
            z_a, c_a, attn_a = Dual_attention(z, index_swap)
            loss_guid = Guide_loss(c_a, z_a)
            re_loss2 = Cross_contrast_loss(z, z_a)

            loss = loss_para[0]*(re_loss+re_loss2)+loss_para[1]*(loss_guid)

            optimizer_Streaming.zero_grad()
            loss.backward()
            optimizer_Streaming.step()
            tot_loss = float(loss + tot_loss)
            Proto_total = Proto_total + c_a.tolist()
            Atten_total = Atten_total + attn_a.tolist()
        Proto_total = torch.tensor(Proto_total)
        Atten_total = torch.tensor(Atten_total)
        Buffer[0] = Proto_total
        Buffer[1] = Atten_total

        if epoch+1 == 1 or (epoch + 1) % 200 == 0:
            with torch.no_grad():
                Dual_attention_G = dual_attention(dim=config.feature_dim, num=(config.Knowledge_size * (batch_idx + 1))).cuda()

                Xv = torch.tensor(Xv).cuda()
                fea_emb, Xr = decoder_model(Xv)

                Z_fea, P_fea, A_fea = Dual_attention_G(fea_emb, index_swap)
                Labels = Y[0]
                estimator = KMeans(n_clusters=cluster)
                fea_cluster = Z_fea.detach().cpu().numpy()
                estimator.fit(fea_cluster)
                pred_final = estimator.labels_

                acc, nmi, purity, fscore, precision, recall, ari = evaluate(Labels, pred_final)
                # print("v=", v, "epoch=", epoch + 1, "acc=", acc * 100, "nmi=", nmi * 100, "fscore=", fscore * 100,
                #       "ari=", ari * 100,
                #       "recall=", recall * 100, "precision=", precision * 100, "loss=", tot_loss)
                #
                # list = [v, epoch + 1, acc * 100, nmi * 100, fscore * 100, ari * 100, recall * 100, precision * 100,
                #         tot_loss]
                # data = pd.DataFrame([list])
                # data.to_csv(path + filename, mode='a', header=False, index=False)

    return Z_fea, Buffer




def train_view(decoder_model, Dual_attention, optimizer_Streaming, Xv, Y, v, path, filename, config, cluster, Buffer, loss_para):
    seed = 0
    np.random.seed(seed)
    rn.seed(seed)

    t_progress = tqdm(range(config.epochs), desc=f'Streaming_View{v}_training')

    for epoch in t_progress:
        index_swap = None
        tot_loss = 0.0

        Proto_total = []
        Atten_total = []
        Rec_loss = Loss_Reconstruction().cuda()
        Guide_loss = Loss_Konw_Guide().cuda()
        Cross_contrast_loss = Cross_contrast().cuda()


        for xs, ys, p_BF, a_BF, batch_No in next_batch(Xv, Y, Buffer[0], Buffer[1], config.batch_size, config.Knowledge_size):

            xs = torch.tensor(xs).cuda()
            p_BF = torch.tensor(p_BF).cuda()
            a_BF = torch.tensor(a_BF).cuda()
            decoder_model = decoder_model.cuda()
            Dual_attention = Dual_attention.cuda()
            xs = xs.squeeze(0)
            z, xr = decoder_model(xs)
            re_loss1 = Rec_loss(xs, xr)
            z = z.squeeze(0)
            z_a, p_a, attn_a = Dual_attention(z, index_swap)

            # re_loss2 = Cross_contrast_loss(z_a, z)
            loss_align = Cross_contrast_loss(p_BF, p_a)

            loss_guid = Guide_loss(p_a, z)

            Recloss = re_loss1
            Alignloss = loss_align + loss_para[1]*loss_guid
            loss = loss_para[0]*Recloss + Alignloss


            # loss = 0.1*re_loss + re_loss2
            optimizer_Streaming.zero_grad()
            loss.backward()
            optimizer_Streaming.step()
            tot_loss = float(loss + tot_loss)
            Proto_total = Proto_total + p_a.tolist()
            Atten_total = Atten_total + attn_a.tolist()
        Proto_total = torch.tensor(Proto_total)
        Atten_total = torch.tensor(Atten_total)
        Buffer[0] = Proto_total
        Buffer[1] = Atten_total



        if epoch+1 == 1 or (epoch + 1) % 200 == 0:
            with torch.no_grad():
                Dual_attention_G = dual_attention(dim=config.feature_dim, num=(config.Knowledge_size * (batch_No))).cuda()
                Buffer[0] = torch.tensor(Buffer[0]).cuda()
                Xv = torch.tensor(Xv).cuda()
                fea_emb, Xr = decoder_model(Xv)

                Z_fea, P_fea, A_fea = Dual_attention_G(fea_emb, index_swap)

                Labels = Y[0]
                estimator = KMeans(n_clusters=cluster)
                fea_cluster = Z_fea.detach().cpu().numpy()
                estimator.fit(fea_cluster)
                pred_final = estimator.labels_

                acc, nmi, purity, fscore, precision, recall, ari = evaluate(Labels, pred_final)
                # print("v=", v, "epoch=", epoch + 1, "acc=", acc * 100, "nmi=", nmi * 100, "fscore=", fscore * 100,
                #       "ari=", ari * 100,
                #       "recall=", recall * 100, "precision=", precision * 100, "loss=", tot_loss)
                #
                # list = [v, epoch + 1, acc * 100, nmi * 100, fscore * 100, ari * 100, recall * 100, precision * 100,
                #         tot_loss]
                # data = pd.DataFrame([list])
                # data.to_csv(path + filename, mode='a', header=False, index=False)

    return Z_fea, Buffer
