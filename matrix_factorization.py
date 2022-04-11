import torch
import utils
from training_pipelines import train_recommendation_model
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()
        # Add +1 to embeddings dimensionality because of indexing
        self.user_embedding = torch.nn.Embedding(num_users + 1, embedding_dim, sparse=sparse)
        self.user_bias = torch.nn.Embedding(num_users + 1, 1, sparse=sparse)

        self.item_embedding = torch.nn.Embedding(num_items + 1, embedding_dim, sparse=sparse)
        self.item_bias = torch.nn.Embedding(num_items + 1, 1, sparse=sparse)

        # Initialize weights from xavier distr
        for param in self.parameters():
            torch.nn.init.xavier_normal_(param)

    def forward(self, user_id, item_id):
        P = self.user_embedding(user_id)
        P_bias = self.user_bias(user_id).flatten()
        Q = self.item_embedding(item_id)
        Q_bias = self.item_bias(item_id).flatten()

        return (P * Q).sum(-1) + P_bias + Q_bias


if __name__ == '__main__':
    torch.manual_seed(42)

    dataset, users_cnt, items_cnt = utils.load_dataset("ml_1m")
    train_df, test_df = utils.split_data(dataset, shuffle=True)

    BATCH_SIZE = 2048
    EMBEDDING_DIM = 10
    EPOCHS = 25
    TARGET = "rating"
    LR = 1
    scheduler = ReduceLROnPlateau
    scheduler_conf = {
        "mode": "min",
        "factor": 0.25,
        "patience": 5,
        "threshold": 5e-2,
    }

    train_iter = utils.create_data_loader(train_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                          item_col="item_id", user_col="user_id")
    test_iter = utils.create_data_loader(test_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                         item_col="item_id", user_col="user_id", train=True)

    mf_net = MatrixFactorization(users_cnt, items_cnt, EMBEDDING_DIM)

    train_recommendation_model(mf_net, train_iter, test_iter, EPOCHS, learning_rate=LR,
                               scheduler=scheduler, scheduler_conf=scheduler_conf, use_tensorboard=True)
