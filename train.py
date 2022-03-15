# add command line params
# --lr: learning rate 0.001 (default: 0.01)
# --random_seed: 42 (default: None)
# --n_epochs: 150 (default: 10)
# --tensorboard (default: False)
# --tensorboard_dir (default: 'run/')

import argparse
import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core.gs_wrappers import SymbolGameGS, SymbolReceiverWrapper, GumbelSoftmaxWrapper

from data import ScaledDataset, enumerate_attribute_value, one_hotify, split_train_test
from architectures import Sender, Receiver


def get_params(params):
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--n_summands", type=int, default=2, help="Number of summands")
    parser.add_argument("--N", type=int, default=20, help="Maximal value of summands")
    parser.add_argument("--test_split", type=float, default=0.1, help="proportion of test set samples")
    parser.add_argument("--data_scaling", type=int, default=50, help="number of occurrences of training samples")
    parser.add_argument("--one_hot", type=bool, default=False, help="whether data is one-hot encoded"),
    # agents and game
    parser.add_argument("--receiver_embed_dim", type=int, default=100,
                        help="embedding dimension for generated symbol")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_symbols", type=int, default=100, help="number of symbols")
    # training
    parser.add_argument("--temperature", type=float, default=1.2, help="GS temperature for the sender")
    parser.add_argument("--temp_decay", type=float, default=0.995, help="temperature decay")
    parser.add_argument("--early_stopping_acc", type=float, default=0.99, help="accuracy for early stopping")
    parser.add_argument("--save_interactions", default=False, action="store_true", help="whether to save interactions")
    parser.add_argument("--n_runs", type=int, default=1, help="number of runs")
    args = core.init(parser, params)
    return args


def run(opts, save_dir):

    print(opts, flush=True)

    latest_run = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, str(latest_run)) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if opts.tensorboard:
        opts.tensorboard_dir = save_path + 'checkpoints/'

    pickle.dump(opts, open(save_path + 'params.pkl', 'wb'))

    def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
        # cross-entropy loss between true sum and receiver output
        if opts.one_hot:
            labels = torch.zeros(sender_input.shape[0], dtype=torch.long).to(opts.device)
            for summand in range(opts.n_summands):
                labels += torch.argmax(sender_input[:, summand*opts.N: (summand+1)*opts.N], dim=1)
        else:
            labels = torch.sum(sender_input, dim=1).long()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        return loss, {"acc": acc}

    # create sender and receiver, wrap for single symbol and gumbel softmax

    if opts.one_hot:
        n_features = opts.N * opts.n_summands
    else:
        n_features = opts.n_summands
    n_sums = opts.N * opts.n_summands
    hidden_dim = 2 * opts.receiver_embed_dim

    sender = Sender(n_features, opts.n_symbols, opts.n_layers, hidden_dim)
    receiver = Receiver(opts.receiver_embed_dim, n_sums, opts.n_layers, hidden_dim)

    sender = GumbelSoftmaxWrapper(sender, temperature=opts.temperature)
    receiver = SymbolReceiverWrapper(receiver, opts.n_symbols, opts.receiver_embed_dim)

    # create game

    game = SymbolGameGS(sender, receiver, loss)

    # generate data

    full_data = enumerate_attribute_value(opts.n_summands, opts.N)

    train, test = split_train_test(full_data, p_hold_out=opts.test_split)
    if opts.one_hot:
        train, test = [one_hotify(x, opts.n_summands, opts.N) for x in [train, test]]
    else:
        train, test = (torch.Tensor(train), torch.Tensor(test))

    train, validation, test, full_data = (ScaledDataset(train, opts.data_scaling),
                                          ScaledDataset(train, 1),
                                          ScaledDataset(test, 1),
                                          ScaledDataset(full_data, 1))

    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=len(validation))

    # define training options

    optimizer = core.build_optimizer(game.parameters())

    callbacks = [core.ConsoleLogger(print_train_loss=True),
                 core.TemperatureUpdater(agent=sender, decay=opts.temp_decay, minimum=0.75),
                 core.EarlyStopperAccuracy(opts.early_stopping_acc)]
    if opts.save_interactions:
        callbacks.append(core.callbacks.InteractionSaver(train_epochs=[opts.n_epochs],
                                                         test_epochs=[opts.n_epochs],
                                                         checkpoint_dir=save_path))

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks
        )

    # train

    trainer.train(n_epochs=opts.n_epochs)

    # test

    test_loader = DataLoader(test, batch_size=opts.batch_size)
    test_loss, test_interaction = trainer.eval(data=test_loader)
    if opts.save_interactions:
        core.InteractionSaver.dump_interactions(test_interaction,
                                                dump_dir=save_path+'interactions/',
                                                mode='test',
                                                epoch=opts.n_epochs,
                                                rank=0)


def main(params):

    opts = get_params(params)
    print(opts, flush=True)

    save_dir = str('results/N' + str(opts.N) + '_vocab-size' + str(opts.n_symbols) + '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(opts.n_runs):
        run(opts, save_dir=save_dir)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
