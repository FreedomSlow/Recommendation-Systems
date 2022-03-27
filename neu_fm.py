import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchmetrics

import utils


class NeuFm(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_dim: list):
        """
        NeuFm consists of two parts: generalized matrix factorization (GMF) and MLP with user-item concatenations
        Then outputs of last two layers in both structures are concatenated and passed through final
        Fully-Connected layer to predict rating

        :param num_users: Number of users in dataset
        :param num_items: Number of items in dataset
        :param embedding_dim: Dimension of embeddings used in model
        :param hidden_dim (List): List of dimensions of hidden layers in MLP
        """
        super(NeuFm, self).__init__()

        # Embeddings
        self.gmf_user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.mlp_user_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.mlp_item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        # MLP
        _mlp = [
            torch.nn.Linear(embedding_dim * 2, hidden_dim[0]),
            torch.nn.LeakyReLU()
        ]

        for dim in range(len(hidden_dim) - 1):
            _mlp += [
                torch.nn.Linear(hidden_dim[dim], hidden_dim[dim + 1]),
                torch.nn.LeakyReLU()
            ]

        self.mlp = torch.nn.Sequential(*_mlp)
        self.output_layer = torch.nn.Linear(
            hidden_dim[-1] + embedding_dim, 1, bias=False
        )

    def forward(self, user_id, item_id):
        p_mf = self.gmf_item_embedding(user_id)
        q_mf = self.gmf_item_embedding(item_id)
        gmf = p_mf * q_mf

        p_mlp = self.mlp_user_embedding(user_id)
        q_mlp = self.mlp_item_embedding(item_id)
        mlp = self.mlp(torch.concat((p_mlp, q_mlp), axis=-1))

        return self.output_layer(torch.concat((gmf, mlp), axis=-1))


class DatasetWithNegativeSampling(Dataset):
    """
    We create new Dataset class because for pairwise ranking loss, an important step is negative sampling.
    For each user, the items that a user has not interacted with are candidate items (unobserved entries).
    The following function takes users identity and candidate items as input,
    and samples negative items randomly for each user from the candidate set of that user.
    During the training stage, the model ensures that the items that a user likes to be ranked higher
    than items he dislikes or has not interacted with.

    For further explanation see https://arxiv.org/pdf/1708.05031.pdf
    """
    def __init__(self, train_df, test_df, target_col, user_col="user_id", item_col="item_id",
                 train=False, items_for_testing_per_user=2):
        super(DatasetWithNegativeSampling, self).__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.user_col = user_col
        self.item_col = item_col
        self.all_items = set(train_df[item_col].values).union(set(test_df[item_col].values))
        self.all_users = set(train_df[user_col].values).union(set(test_df[user_col].values))
        self.train = train
        self.items_for_testing_per_user = items_for_testing_per_user
        self.generate_candidates()

    def generate_candidates(self):
        # Items user interacted with
        self.interacted_during_training = {
            int(user_id): set(user_df[self.item_col].values)
            for user_id, user_df in self.train_df.groupby(self.user_col)
        }

        # Items user did not interact with
        self.unobserved_during_training = {
            user_id: np.array(
                set(self.all_items) - observed
            )
            for user_id, observed in self.interacted_during_training.items()
        }

        self.gt_pos_items_per_user_in_test = {
            int(user_id): user_df[-self.items_for_testing_per_user:].item_id.values
            for user_id, user_df in self.test_df.groupby("user_id")
        }

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, idx):
        """Get negative candidate for user_id"""
        if self.train:
            user_id = self.train_df[self.user_col].values[idx]
            pos_item = self.train_df[self.item_col].values[idx]
            neg_item = np.random.choice(
                self.unobserved_during_training[int(user_id)])
            return user_id, pos_item, neg_item
        else:
            user_id = self.test_df[self.user_col].values[idx]
            item_id = self.test_df[self.item_col].values[idx]
            is_pos = item_id in self.gt_pos_items_per_user_in_test[user_id]
            return user_id, item_id, is_pos


def train_ranking_model(net, train_iter, test_iter, epochs, learning_rate=1e-4, loss=None,
                        device=None, save_optim=None, **kwargs):
    """
    Train pairwise ranking model using user-item interactions positive examples
    and using unobserved items as negative examples.
    :param net:
    :param train_iter:
    :param test_iter:
    :param epochs:
    :param learning_rate:
    :param loss:
    :param device:
    :param save_optim:
    :param kwargs:
    :return:
    """

    plotter = utils.PlotResults(x_label="epoch", x_lim=[1, epochs], y_lim=[0, 1],
                                legend=["hit rate"])

    device = utils.try_gpu() if device is None else device
    print(f"Training model on: {device}")
    net.to(device)

    if loss is None:
        if "margin" in kwargs:
            _hinge_margin = kwargs["margin"]
        loss = utils.loss_functions.hinge_loss_rec

    # Pytorch Embeddings work only with SGD (CPU/GPU), Adagrad (CPU)
    optimizer = torch.optim.SGD(lr=learning_rate, **kwargs)

    hit_rate = 0

    for epoch in range(epochs):
        metrics = utils.SaveMetrics(2)
        # Set gradients to train mode
        net.train()

        for i, (X, y) in enumerate(tqdm(train_iter)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)


if __name__ == '__main__':
    pass
