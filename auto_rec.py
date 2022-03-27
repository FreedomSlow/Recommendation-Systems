import torch
import numpy as np
import utils


class AutoRec(torch.nn.Module):
    def __init__(self, num_input, num_hidden, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_input, num_hidden),
            torch.nn.LogSigmoid(),
            torch.nn.Dropout(dropout)
        )
        self.decoder = torch.nn.Linear(num_hidden, num_input)

        # Initialize weights with Xavier distribution
        for param in self.parameters():
            torch.nn.init.xavier_normal_(param)

    def forward(self, input_data):
        return self.decoder(self.encoder(input_data))


# TODO: Write evaluator function


if __name__ == '__main__':
    dataset, users_cnt, items_cnt = utils.load_dataset("ml_small")
    train_df, test_df = utils.split_data(dataset, shuffle=False)

    BATCH_SIZE = 32
    EMBEDDING_DIM = 512
    EPOCHS = 10
    TARGET = "rating"

    train_iter = utils.create_data_loader(train_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                          item_col="item_id", user_col="user_id")
    test_iter = utils.create_data_loader(test_df, batch_size=BATCH_SIZE, target_col=TARGET,
                                         item_col="item_id", user_col="user_id")

    # Since we are making item-based AutoRec input dimension will be number of users
    # That way our model will learn complete predictions for all missing values
    ar_net = AutoRec(users_cnt, EMBEDDING_DIM)

    utils.train_recommendation_model(ar_net, train_iter, test_iter, EPOCHS, learning_rate=1e-2)
