import torch
from tqdm import tqdm
import utils


class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()
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


def train_recommendation_model(net, train_iter, val_iter, epochs, learning_rate=1e-4, loss=None,
                               device=None, save_optim=None, **kwargs):

    """
    Train simple recommendation model using user-item matrix to predict rating for all unseen items
    :param net: Torch model
    :param train_iter: Torch train DataLoader with X tensor of (user_id, item_id) and y tensor of their ratings
    :param val_iter: Same as train_iter but for validation
    :param epochs: Number of epochs
    :param learning_rate: Learning rate
    :param loss: Loss function
    :param device: Device to train model on
    :param save_optim: Either to save optimizer state
    :param kwargs:
    """

    plotter = utils.PlotResults(x_label="epoch", y_label="RMSE loss", x_lim=[1, epochs], y_lim=[0, 2],
                                legend=["train loss", "test loss"])
    if device is None:
        device = utils.try_gpu()
    print(f"Training model on: {device}")
    net.to(device)
    # Pytorch Embeddings work only with SGD (CPU/GPU), Adagrad (CPU)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    if loss is None:
        loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        metrics = utils.SaveMetrics(2)
        # Set gradients to train mode
        net.train()

        for i, (X, y) in enumerate(tqdm(train_iter)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            user_tensor = utils.get_id_from_tensor(X, "user")
            item_tensor = utils.get_id_from_tensor(X, "item")

            y_pred = net(user_tensor, item_tensor)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metrics.add(l * X.shape[0], X.shape[0])

            train_loss = metrics[0] / metrics[1]

        print(f"epoch: {epoch}", f"train loss: {train_loss:.3f}")

    if save_optim is not None:
        torch.save(optimizer.state_dict(), save_optim)
        return f"Optimizer saved to {save_optim}"

    return


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

    train_recommendation_model(mf_net, train_iter, test_iter, EPOCHS, learning_rate=1e-2)
