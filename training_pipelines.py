import torch
from tqdm import tqdm
import utils
from torch.utils.tensorboard import SummaryWriter
import loss_functions
import torchmetrics


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