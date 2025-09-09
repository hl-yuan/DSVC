import scipy.io as sio
import h5py
import torch
import random
import warnings
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")


# path = './datasets/'

ALL_data = dict(
    HandWritten={1: 'HandWritten', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [240, 76, 216, 47, 64, 6], "Knowledge_size": (7*10), "beta1": 1e-0, "beta2": 1e-1},
    Scene_15={1: 'Scene_15', 'N': 4485, 'K': 15, 'V': 3, 'n_input': [20, 59, 40], "Knowledge_size": (15), "beta1": 1e-0,"beta2": 1e-1},
    YoutubeFace_sel_fea={1: 'YoutubeFace_sel_fea', 'N': 101499, 'K': 31, 'V': 5, 'n_input': [64, 512, 64, 647, 838], "Knowledge_size": 31, "beta1": 1e-0, "beta2": 1e-1},
    stl10_fea={1: 'stl10_fea', 'N': 13000, 'K': 10, 'V': 3, 'n_input': [1024,512,2048], "Knowledge_size": (10), "beta1": 1e-3, "beta2": 1e-0},
    LandUse_21={1: 'LandUse_21', 'N': 2100, 'K': 21, 'V': 3, 'n_input': [20, 59, 40], "Knowledge_size": (1*21), "beta1": 0.1, "beta2": 0.1},
    ALOI_10={1: 'ALOI_10', 'N': 1079, 'K': 10, 'V': 4, 'n_input': [64, 64, 77, 13], "Knowledge_size": 1*10, "beta1": 1e-0, "beta2": 1e-1},
    ALOI_100={1: 'ALOI_100', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125], "Knowledge_size": 100, "beta1": 1e-1, "beta2": 1e-1},
    ALOI_1000={1: 'ALOI_1000', 'N': 110250, 'K': 1000, 'V': 4, 'n_input': [125, 77, 13, 64], "Knowledge_size": 10, "beta1": 1e-1, "beta2": 1e-1},
)
path = './Datasets/'


"""other read"""
def load_data2(dataset):
    if dataset[1]=='YoutubeFace_sel_fea':
        X, Y = load_datav4(dataset)
    elif dataset[1]=='stl10_fea':
        X, Y = load_datav3(dataset)
    elif dataset[1]=='Scene_15':
        X, Y = load_datav3(dataset)
    else:
        X, Y = load_datav1(dataset)
    return X, Y





def load_datav2(dataset):
    file_path = os.path.join(path, dataset[1])
    mat_data = sio.loadmat(file_path)
    X_raw = mat_data['X']
    X_raw = X_raw.T
    Y_raw = mat_data['Y']
    Label = np.array(Y_raw).reshape(-1)
    X = []
    mm = MinMaxScaler()
    nn = mat_data['X'].shape[0]
    print(nn)
    for i in range(dataset['V']):
        view_data = np.array(X_raw[0, i], dtype=np.float32)

        view_data = mm.fit_transform(view_data)  # 归一化
        X.append(view_data)
    Y = [Label for _ in range(len(X))]
    return X, Y

def load_datav1(dataset):
    file_path = os.path.join(path, dataset[1])
    mat_data = sio.loadmat(file_path)
    X_raw = mat_data['X']
    Y_raw = mat_data['Y']
    Label = np.array(Y_raw).reshape(-1)

    X = []
    mm = MinMaxScaler()

    for i in range(dataset['V']):
        from scipy.sparse import csc_matrix
        if isinstance(X_raw[0, i], csc_matrix):
            view_data = np.array(X_raw[0, i].toarray(), dtype=np.float32)
        else:
            view_data = np.array(X_raw[0, i], dtype=np.float32)

        view_data = mm.fit_transform(view_data)  # 归一化
        X.append(view_data)
    Y = [Label for _ in range(len(X))]
    return X, Y

def load_datav4(dataset):
    file_path = os.path.join(path, dataset[1])
    mat_data = sio.loadmat(file_path)
    X_raw = mat_data['X']
    Y_raw = mat_data['Y']
    Label = np.array(Y_raw).reshape(-1)

    X = []
    mm = MinMaxScaler()

    for i in range(dataset['V']):
        if dataset[1] == "YoutubeFace_sel_fea":
            view_data = np.array(X_raw[i, 0], dtype=np.float32).T
        else:
            view_data = np.array(X_raw[i, 0], dtype=np.float32)
        print(view_data.shape)
        view_data = mm.fit_transform(view_data)
        X.append(view_data)
    Y = [Label for _ in range(len(X))]
    return X, Y


"""v73 read"""
def load_datav3(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T
    print(data['X'])
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(dataset['V']):
        if dataset[1] == 'Scene_15' or dataset[1] == 'AwA30475':
            diff_view = data[data['X'][0, i]]
        else:
            diff_view = data[data['X'][i, 0]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
    Y.append(Label)
    return X, Y


def load_datav5(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['gt']).T
    print(data['fea'])
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(dataset['V']):
        diff_view = data[data['fea'][i, 0]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
    Y.append(Label)
    return X, Y

def load_datav6(dataset):
    file_path = os.path.join(path, dataset[1])
    mat_data = sio.loadmat(file_path)
    X_raw = mat_data['data']
    Y_raw = mat_data['truelabel'][0, 0]
    Label = np.array(Y_raw).reshape(-1)

    X = []
    mm = MinMaxScaler()
    # nn = mat_data['X'].shape[1]
    for i in range(dataset['V']):
        view_data = np.array(X_raw[i, 0], dtype=np.float32).T
        print(view_data.shape)
        view_data = mm.fit_transform(view_data)  # 归一化
        X.append(view_data)
    Y = [Label for _ in range(len(X))]
    return X, Y

class TrainDataset_Com(torch.utils.data.Dataset):
    def __init__(self, X_list, Y_list):
        self.X_list = X_list
        self.Y_list = Y_list
        # self.Z_list = Z_list

    def __getitem__(self, index):
        current_x_list = []
        current_y_list = []
        current_x = self.X_list[index]
        current_x_list.append(current_x)
        current_y = self.Y_list
        current_y_list.append(current_y)
        return current_x_list, current_y_list

    def __len__(self):
        return self.X_list.shape[0]

class Data_Sampler(object):

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

