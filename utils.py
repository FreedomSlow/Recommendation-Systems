import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import train_test_split
import torchmetrics
import loss_functions
from tqdm import tqdm


#######################################################
#                 Utility Functions                   #
#######################################################

def try_gpu():
    """Select torch cuda device"""
    if torch.cuda.device_count() > 0:
        return torch.device("cuda:0")

    return torch.device("cpu")


def load_dataset(dataset_name):
    """Load dataset into pandas.DataFrame and return number of unique users and items"""
    _names = ["user_id", "item_id", "rating", "timestamp"]

    if dataset_name == "ml_100k":
        # Set column names
        dataset = pd.read_csv("datasets/ml-100k/u.data", sep="\t", engine="python", names=_names)

    elif dataset_name == "ml_small":
        dataset = pd.read_csv("datasets/ml-latest-small/ratings.csv", sep=",", skiprows=0)
        dataset.columns = _names

    elif dataset_name == "ml_1m":
        dataset = pd.read_csv("datasets/ml-1m/ratings.dat", sep="::", names=_names, engine="python")

    else:
        raise ValueError(f"dataset_name must be either 'ml_100k', 'ml_small' or 'ml-1m' got {dataset_name},"
                         f" please check datasets for further explanation")

    # Get max of every id because they can be non-monotonous
    _users_cnt = dataset["user_id"].max()
    _items_cnt = dataset["item_id"].max()

    return dataset, _users_cnt, _items_cnt


def split_data(df, shuffle=True, train_size=None, validation_df=False):
    """
    Split data into train, validation (optional) and test dataFrames
    Validation df size is fixed to 0.15 of train df
    """

    train_size = 0.85 if train_size is None else train_size

    _train_df, _test_df = train_test_split(df, shuffle=shuffle, train_size=train_size)
    # return train_test_split(df, shuffle=shuffle, train_size=train_size)

    if validation_df:
        _train_df, _val_df = train_test_split(df, shuffle=False, train_size=train_size)
        return _train_df, _val_df, _test_df

    return _train_df, _test_df


def create_data_loader(df, batch_size, user_col, item_col, target_col, shuffle=True, train=False):
    """Create torch.DataLoaders where from pandas.DataFrame"""
    _X_tensor = torch.IntTensor(df[[user_col, item_col]].values)
    _y_tensor = torch.Tensor(df[target_col].values)
    if train:
        return _X_tensor, _y_tensor

    torch_dataset = TensorDataset(_X_tensor, _y_tensor)

    return DataLoader(torch_dataset, batch_size, shuffle=shuffle)


def get_id_from_tensor(tensor, id_name):
    """Gets appropriate id column from tensor with ids"""
    if id_name == "user":
        return tensor[:, 0]

    elif id_name == "item":
        return tensor[:, 1]

    else:
        raise ValueError(f"id_name should be one of 'user', 'item', got {id_name}")


#######################################################
# Visualization (Deprecated, use tensorboard instead) #
#######################################################


class SaveMetrics:
    """Save metrics during training and validation for n variables"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PlotResults:
    """Class for visualization training and validation"""

    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib."""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def __init__(self, x_label=None, y_label=None, legend=None, x_lim=None, y_lim=None,
                 x_scale="linear", y_scale="linear", linestyles=('-', 'm--', 'g-.', 'r:'),
                 n_rows=1, n_cols=1, figsize=(8, 6)):

        # Incrementally plot graphs during training
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            self.axes = [self.axes,]
        # Use lambda to set arguments
        self.config_axes = lambda: self.set_axes(self.axes[0],
                                                 x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend)
        self.X, self.Y, self.fmts = None, None, linestyles

    def add(self, x, y):
        # Add line to graph
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        self.fig.show()





