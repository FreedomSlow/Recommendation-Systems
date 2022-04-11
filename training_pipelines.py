import torch
from tqdm import tqdm
import utils
import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def train_recommendation_model(net, train_iter, val_iter, epochs, learning_rate=1e-4, loss=None,
                               device=None, save_optim=None, scheduler=None, scheduler_conf=None,
                               use_tensorboard=False, **kwargs):

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

    if use_tensorboard is not None:
        writer = SummaryWriter()

    if device is None:
        device = utils.try_gpu()
    print(f"Training model on: {device}")
    net.to(device)
    # Pytorch Embeddings work only with SGD (CPU/GPU), Adagrad (CPU)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    if scheduler is not None:
        if scheduler_conf is None:
            _scheduler = scheduler(optimizer)
        else:
            _scheduler = scheduler(optimizer, **scheduler_conf)
    if loss is None:
        loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        # Set gradients to train mode
        net.train()

        for i, (X, y) in enumerate(tqdm(train_iter)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            user_tensor = utils.get_id_from_tensor(X, "user")
            item_tensor = utils.get_id_from_tensor(X, "item")

            y_pred = torch.flatten(net(user_tensor, item_tensor))
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():

                X_test, y_test = val_iter
                X_test, y_test = X_test.to(device), y_test.to(device)
                _test_user_tensor = utils.get_id_from_tensor(X_test, "user")
                _test_item_tensor = utils.get_id_from_tensor(X_test, "item")

                y_test_pred = torch.flatten(net(_test_user_tensor, _test_item_tensor))

                test_loss = loss(y_test_pred, y_test)
                test_rmse = torchmetrics.functional.mean_squared_error(y_test_pred, y_test, squared=False)

                # Add train and test loss to Tensorboard
                if use_tensorboard is not None:
                    writer.add_scalar("train_loss", l, epoch)
                    writer.add_scalar("test_loss", test_loss, epoch)
                    writer.add_scalar("test_RMSE", test_rmse, epoch)

        print(f"epoch: {epoch}", f"train loss: {l:.3f}", f"test loss: {test_loss} test RMSE: {test_rmse}")

    if save_optim is not None:
        torch.save(optimizer.state_dict(), save_optim)
        return f"Optimizer saved to {save_optim}"

    return
