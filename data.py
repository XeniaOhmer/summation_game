# data generation process copied over and adapted from:
# https://github.com/facebookresearch/EGG/blob/main/egg/zoo/compo_vs_generalization/data.py


import itertools
import torch
import numpy as np
from torch.utils.data import Dataset


def enumerate_attribute_value(n_attributes, n_values):
    iters = [range(n_values) for _ in range(n_attributes)]
    return list(itertools.product(*iters))


def one_hotify(data, n_attributes, n_values):
    r = []
    for config in data:
        z = torch.zeros((n_attributes, n_values))
        for i in range(n_attributes):
            z[i, config[i]] = 1
        r.append(z.view(-1))
    return r


def split_train_test(dataset, p_hold_out=0.1, random_seed=7):

    assert p_hold_out > 0
    random_state = np.random.RandomState(seed=random_seed)

    n = len(dataset)
    permutation = random_state.permutation(n)

    n_test = int(p_hold_out * n)

    test = [dataset[i] for i in permutation[:n_test]]
    train = [dataset[i] for i in permutation[n_test:]]
    assert train and test

    assert len(train) + len(test) == len(dataset)

    return train, test


def split_train_test_generalization(dataset, N):

    train, hold_out = [], []
    holdout_values_s1 = list(range(5, N, 20))
    holdout_values_s2 = list(range(15, N, 20))

    for values in dataset:
        if values[0] in holdout_values_s1 or values[1] in holdout_values_s2:
            hold_out.append(values)
        else:
            train.append(values)
    return train, hold_out


class ScaledDataset(Dataset):
    def __init__(self, examples, scaling_factor=1):
        self.examples = examples
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.examples) * self.scaling_factor

    def __getitem__(self, k):
        k = k % len(self.examples)
        return self.examples[k], torch.zeros(1)


if __name__ == "__main__":

    dataset = enumerate_attribute_value(n_attributes=2, n_values=10)
    train, test = split_train_test(dataset)
    print(len(train), len(test), len(dataset))

    print([x[0] for x in [train, test, dataset]])
