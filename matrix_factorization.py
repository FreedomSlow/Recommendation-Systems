import torch
import utils


class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()
        # super(MatrixFactorization, self).__init__(**kwargs)
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim, sparse=sparse)
        self.user_bias = torch.nn.Embedding(num_users, 1, sparse=sparse)

        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim, sparse=sparse)
        self.item_bias = torch.nn.Embedding(num_items, 1, sparse=sparse)

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
    dataset, users_cnt, items_cnt = utils.load_dataset("ml_small")
    train_df, test_df = utils.split_data(dataset, shuffle=False)

    BATCH_SIZE = 32
    EMBEDDING_DIM = 3
    EPOCHS = 10
    TARGET = "rating"

    train_iter = utils.create_data_loader(train_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                          item_col="item_id", user_col="user_id")
    test_iter = utils.create_data_loader(test_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                         item_col="item_id", user_col="user_id")

    mf_net = MatrixFactorization(users_cnt, items_cnt, EMBEDDING_DIM)

    utils.train_recommendation_model(mf_net, train_iter, test_iter, EPOCHS, learning_rate=1e-2)
