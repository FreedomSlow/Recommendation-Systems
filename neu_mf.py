import torch
from training_pipelines import train_recommendation_model
import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NeuMf(torch.nn.Module):
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
        super(NeuMf, self).__init__()

        # Embeddings
        self.gmf_user_embedding = torch.nn.Embedding(num_users + 1, embedding_dim)
        self.gmf_item_embedding = torch.nn.Embedding(num_items + 1, embedding_dim)
        self.mlp_user_embedding = torch.nn.Embedding(num_users + 1, embedding_dim)
        self.mlp_item_embedding = torch.nn.Embedding(num_items + 1, embedding_dim)

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
        self.logit = torch.nn.Sigmoid()

    def forward(self, user_id, item_id):
        p_mf = self.gmf_user_embedding(user_id)
        q_mf = self.gmf_item_embedding(item_id)
        gmf = p_mf * q_mf

        p_mlp = self.mlp_user_embedding(user_id)
        q_mlp = self.mlp_item_embedding(item_id)
        mlp = self.mlp(torch.concat((p_mlp, q_mlp), axis=-1))

        return self.output_layer(torch.concat((gmf, mlp), axis=-1))


if __name__ == '__main__':
    torch.manual_seed(42)

    dataset, users_cnt, items_cnt = utils.load_dataset("ml_1m")
    train_df, test_df = utils.split_data(dataset, shuffle=True)

    BATCH_SIZE = 8192
    EMBEDDING_DIM = 10
    EPOCHS = 50
    HIDDEN_DIMS = [8, 16, 64, 32, 16, 8]
    TARGET = "rating"
    LR = 2e-2
    # loss_function = torch.nn.BCELoss()
    scheduler = ReduceLROnPlateau
    scheduler_conf = {
        "mode": "min",
        "factor": 0.25,
        "patience": 5,
        "threshold": 5e-2,
        "verbose": True
    }

    train_iter = utils.create_data_loader(train_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                          item_col="item_id", user_col="user_id")
    test_iter = utils.create_data_loader(test_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                         item_col="item_id", user_col="user_id", train=True)

    neu_fm = NeuMf(users_cnt, items_cnt, EMBEDDING_DIM, HIDDEN_DIMS)

    train_recommendation_model(neu_fm, train_iter, test_iter, EPOCHS, LR,
                               scheduler=scheduler, scheduler_conf=scheduler_conf)
