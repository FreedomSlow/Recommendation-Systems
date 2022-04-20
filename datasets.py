import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import utils


class DatasetWithNegativeSampling(Dataset):
    """
    We create new Dataset class because for pairwise ranking loss, an important step is negative sampling.
    For each user, the items that a user has not interacted with are candidate items (unobserved entries).
    The following function takes users identity and candidate items as input,
    and samples negative items randomly for each user from the candidate set of that user.
    During the training stage, the model ensures that the items that a user likes to be ranked higher
    than items he dislikes or has not interacted with.
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


class ImplicitFeedbackDataset(Dataset):
    def __init__(self, user_item_tensor, target_tensor):
        super(ImplicitFeedbackDataset, self).__init__()
        self.user_item_tensor = user_item_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.user_item_tensor)

    def __getitem__(self, index):
        return self.user_item_tensor[index], self.target_tensor[index]


class SetupImplicitDataset:
    """
    Almost always recommendation systems are built based on implicit feedback .
    One way to construct such data is to use items users interacted with as positive examples
    And other items as negative. However, it does not mean that user actually likes item they interacted with,
    Similarly it does not mean that users don't like items they didn't interact.

    This Dataset can handle explicit and implicit feedback simultaneously and designed to be used in training
    Neural networks with BCE loss https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
    So we normalize explicit feedback (ratings) and binarize implicit
    """
    def __init__(self, df, user_col="user_id", item_col="item_id", target_col="rating"):
        self.df = df
        self.user_col = user_col
        self.item_col = item_col
        self.target_col = target_col
        self.all_users = set(self.df[self.user_col].unique())
        self.all_items = set(self.df[self.item_col].unique())

    def binarize_implicit(self, df, num_negatives):
        interacted_items = df.groupby(self.user_col)[self.item_col].unique().to_dict()

        unseen_items = {
            uid: np.random.choice(
                list(self.all_items.difference(set(interacted))), num_negatives
            )
            for uid, interacted in interacted_items.items()
        }

        # Create new df with both positive and negative items
        new_df = pd.DataFrame({
            self.user_col: interacted_items.keys(),
            "pos_items": interacted_items.values(),
            "neg_items": unseen_items.values()
        })
        new_df["pos_items"] = new_df["pos_items"].apply(lambda x: list(x))
        new_df["neg_items"] = new_df["neg_items"].apply(lambda x: list(x))
        new_df[self.item_col] = new_df["pos_items"] + new_df["neg_items"]
        new_df["pos_flag"] = new_df["pos_items"].apply(lambda x: [True for _ in range(len(x))])
        new_df["neg_flag"] = new_df["neg_items"].apply(lambda x: [False for _ in range(len(x))])
        new_df[self.target_col] = new_df["pos_flag"] + new_df["neg_flag"]
        # Drop auxiliary columns
        new_df = new_df.drop(["pos_items", "neg_items", "pos_flag", "neg_flag"], axis=1)
        new_df = new_df.explode([self.item_col, self.target_col])
        # Cast all columns to float32
        for col in new_df.columns:
            new_df[col] = new_df[col].astype("int32")

        return new_df

    def normalize_explicit(self):
        """Normalize rating values into [0; 1] from [0; max_rating]"""
        new_df = self.df.copy()
        max_rating = max(new_df[self.target_col])
        new_df[self.target_col] = new_df[self.target_col] / max_rating
        return new_df[[self.user_col, self.item_col, self.target_col]]

    def setup(self, feedback_type="implicit", num_negatives=100, shuffle=True, train_size=None,
              validation_df=False, as_torch_tensor=True):
        if feedback_type == "implicit":
            new_df = self.binarize_implicit(self.df, num_negatives)
        elif feedback_type == "explicit":
            new_df = self.normalize_explicit()
        else:
            raise ValueError(f"feedback_type should be one of 'implicit', 'explicit', got {feedback_type}")

        # Split the data
        if validation_df:
            train_df, val_df, test_df = utils.split_data(new_df, shuffle, train_size, validation_df)

            if as_torch_tensor:
                _cols = [[self.user_col, self.item_col], self.target_col]
                train_tensor = utils.torch_from_pandas(train_df, cols=_cols)
                val_tensor = utils.torch_from_pandas(val_df, cols=_cols)
                test_tensor = utils.torch_from_pandas(test_df, cols=_cols)
                return train_tensor, val_tensor, test_tensor

            return train_df, val_df, test_df

        else:
            train_df, test_df = utils.split_data(new_df, shuffle, train_size, validation_df)
            if as_torch_tensor:
                _cols = [[self.user_col, self.item_col], self.target_col]
                _types = ["int", "float"]
                train_tensor = utils.torch_from_pandas(train_df, cols=_cols, types=_types)
                test_tensor = utils.torch_from_pandas(test_df, cols=_cols, types=_types)
                return train_tensor, test_tensor

            return train_df, test_df
