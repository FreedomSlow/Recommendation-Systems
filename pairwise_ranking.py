import torch
from torch.utils.data import Dataset
import utils
import numpy as np
import pandas as pd
import loss_functions
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


#######################################################
#              For pairwise ranking                   #
#######################################################

class DatasetWithNegativeSampling(Dataset):
    """
    We create new Dataset class because for pairwise ranking loss, an important step is negative sampling.
    For each user, the items that a user has not interacted with are candidate items (unobserved entries).
    The following function takes users identity and candidate items as input,
    and samples negative items randomly for each user from the candidate set of that user.
    During the training stage, the model ensures that the items that a user likes to be ranked higher
    than items he dislikes or has not interacted with.

    For further explanation check https://arxiv.org/pdf/1708.05031.pdf
    """
    def __init__(self, train_df, test_df, user_col="user_id", item_col="item_id",
                 train=False, training_pairs_per_user=None, num_positive_in_test=2, k=10):
        super(DatasetWithNegativeSampling, self).__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.user_col = user_col
        self.item_col = item_col
        self.all_items = set(train_df[item_col].values).union(set(test_df[item_col].values))
        self.all_users = set(train_df[user_col].values).union(set(test_df[user_col].values))
        self.train = train
        self.training_pairs_per_user = training_pairs_per_user
        self.num_positive_in_test = num_positive_in_test
        # Will be used while generating training and test df
        self.pos_item_col = "pos_item"
        self.neg_item_col = "neg_item"
        self.k = k
        # Run setup methods
        self.generate_candidates()
        self.build_new_df()

    def generate_candidates(self):
        # Items user interacted with
        self.interacted_during_training = {
            int(user_id): set(user_df[self.item_col].values)
            for user_id, user_df in self.train_df.groupby(self.user_col)
        }

        # Items user did not interact with
        self.unobserved_during_training = {
            user_id: np.array(
                list(self.all_items - observed)
            )
            for user_id, observed in self.interacted_during_training.items()
        }

        self.pos_items_per_user_in_test = {
            int(user_id): np.random.choice(
                user_df[self.item_col].values, self.num_positive_in_test
            )
            for user_id, user_df in self.test_df.groupby(self.user_col)
        }

    def build_new_df(self):
        # Build train tensor for DataLoader with three columns: user_id, pos_item_id, neg_item_id
        if self.train:
            # Use all available interacted items for training if None by setting to 0
            if self.training_pairs_per_user is None:
                _users_items_interacted = {
                    uid: len(items)
                    for uid, items in self.interacted_during_training.items()
                }
            else:
                _users_items_interacted = {
                    uid: self.training_pairs_per_user
                    for uid in self.interacted_during_training
                }

            _user_id = self.train_df[self.user_col].unique()
            users = [uid for uid in _user_id for _ in range(_users_items_interacted[uid])]

            _pos_items = np.array([
                    np.random.choice(
                        list(self.interacted_during_training[uid]), _users_items_interacted[uid]
                    ) for uid in _user_id
            ]).flatten()

            _neg_items = np.array([
                    np.random.choice(
                        list(self.unobserved_during_training[uid]), _users_items_interacted[uid]
                    ) for uid in _user_id
            ]).flatten()

            assert len(_neg_items) == len(_pos_items) == len(users)

            new_train_df = pd.DataFrame({
                    self.user_col: users,
                    self.pos_item_col: _pos_items,
                    self.neg_item_col: _neg_items
            })

            self.new_train_df = torch.from_numpy(new_train_df.values)

        # Build test tensor for DataLoader with three columns: user_id, item_id, is_positive
        else:
            _items_for_test = {
                int(user_id): user_df[self.item_col].values
                for user_id, user_df in self.test_df.groupby(self.user_col)
            }

            _users = [uid for uid in _items_for_test for _ in range(self.k)]

            # Because we simulate real interaction feedback we'll manually add at least one
            # Item which will be True in interactions
            # _items = np.array([
            #     np.append(
            #         np.random.choice(
            #             _items_for_test[uid], self.k - 1
            #         ), np.random.choice(self.pos_items_per_user_in_test[uid])
            #     ) for uid in _items_for_test.keys()
            # ]).flatten()

            _items = np.array([
                np.random.choice(
                    _items_for_test[uid], self.k
                ) for uid in _items_for_test.keys()
            ]).flatten()

            _is_positive = np.array([
                item_id in self.pos_items_per_user_in_test[uid]
                for item_id, uid in zip(_items, _users)
            ])

            assert len(_users) == len(_items) == len(_is_positive)

            new_test_df = pd.DataFrame({
                self.user_col: _users,
                self.item_col: _items,
                "is_positive": _is_positive
            })
            new_test_df["is_positive"] = new_test_df["is_positive"].astype(int)

            self.new_test_df = torch.from_numpy(new_test_df.values)

        return

    def __len__(self):
        if self.train:
            return len(self.new_train_df)

        return len(self.new_test_df)

    def __getitem__(self, idx):
        """Get negative candidate for user_id"""
        if self.train:
            user_id, pos_item, neg_item = self.new_train_df[idx]
            return user_id, pos_item, neg_item

        else:
            user_id, item_id, is_pos = self.new_test_df[idx]
            return user_id, item_id, is_pos


def train_ranking_model(net, train_iter, test_iter, epochs, learning_rate=1e-4, loss=None,
                        device=None, save_optim=None, hitrate_k=5, **kwargs):
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

    writer = SummaryWriter()

    device = utils.try_gpu() if device is None else device
    print(f"Training model on: {device}")
    net.to(device)

    if loss is None:
        if "margin" in kwargs:
            _hinge_margin = kwargs["margin"]
        else:
            _hinge_margin = 1
        loss = loss_functions.hinge_loss_rec

    hitrate = torchmetrics.RetrievalHitRate(k=hitrate_k)

    # Pytorch Embeddings work only with SGD (CPU/GPU), Adagrad (CPU)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, **kwargs)

    for epoch in range(epochs):
        # Set gradients to train mode
        net.train()

        # One observation (X matrix) in case of pairwise ranking consists of user_id, positive item_id
        # And negative item_id
        for i, batch in enumerate(tqdm(train_iter)):
            optimizer.zero_grad()

            user_id, pos_item, neg_item = batch
            user_id = user_id.type(torch.IntTensor)
            pos_item = pos_item.type(torch.IntTensor)
            neg_item = neg_item.type(torch.IntTensor)

            y_pred_pos = net(user_id, pos_item)
            y_pred_neg = net(user_id, neg_item)

            l = loss(y_pred_pos, y_pred_neg)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                writer.add_scalar("train_loss", l, epoch)
                hit_rate = 0
                _cnt = 0
                for test_batch in test_iter:
                    test_user_id, test_item_id, test_target = test_batch
                    test_user_id = test_user_id.type(torch.LongTensor)
                    test_item_id = test_item_id.type(torch.IntTensor)
                    test_target = test_target.type(torch.IntTensor)

                    test_pred = torch.flatten(net(test_user_id, test_item_id))

                    hit_rate += hitrate(test_pred, test_target, indexes=test_user_id)
                    _cnt += 1

                hit_rate = hit_rate / _cnt

                writer.add_scalar("test_HitRate", hit_rate, epoch)

        print(f"epoch: {epoch}", f"train loss: {l:.3f}", f"test loss: {hit_rate:.3f}")