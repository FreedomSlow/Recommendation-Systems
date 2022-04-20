from datasets import ImplicitFeedbackDataset, SetupImplicitDataset
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from training_pipelines import train_recommendation_model
import utils


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
    BATCH_SIZE = 8192
    EMBEDDING_DIM = 20
    EPOCHS = 50
    HIDDEN_DIMS = [8, 16, 64, 32, 16, 8]
    TARGET = "rating"
    LR = 2e-2
    # For implicit feedback use BCELoss() and MSELoss() for explicit
    loss_function = torch.nn.BCELoss()
    scheduler = ReduceLROnPlateau
    scheduler_conf = {
        "mode": "min",
        "factor": 0.25,
        "patience": 5,
        "threshold": 5e-2,
        "verbose": True
    }

    dataset, users_cnt, items_cnt = utils.load_dataset("ml_small")
    setuper = SetupImplicitDataset(dataset)
    # SetupImplicitDataset returns list of tensors: user_id-tensor, item_id-tensor and rating-tensor
    _train_data, _test_data = setuper.setup()
    train_tensor = ImplicitFeedbackDataset(*_train_data)
    test_tensor = ImplicitFeedbackDataset(*_test_data)

    train_iter = DataLoader(train_tensor, batch_size=BATCH_SIZE)
    test_iter = DataLoader(test_tensor, batch_size=len(test_tensor))

    neu_fm = NeuMf(users_cnt, items_cnt, EMBEDDING_DIM, HIDDEN_DIMS)

    train_recommendation_model(neu_fm, train_iter, test_iter, EPOCHS, LR,
                               scheduler=scheduler, scheduler_conf=scheduler_conf, feedback_type="implicit")
