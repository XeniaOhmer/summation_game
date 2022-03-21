import os
import torch
import numpy as np
from egg.core.language_analysis import calc_entropy, _hashable_tensor


def load_interaction(N, vocab_size, mode='validation'):
    path = 'results/N' + str(N) + '_vocab-size' + str(vocab_size) + '/interactions/' + mode + '/'
    folder = os.listdir(path)[0]
    logs = torch.load(path + folder + '/interaction_gpu0')
    return logs


def one_hot_to_numbers(inputs):

    N = inputs.shape[1] // 2
    numbers1 = inputs[:, 0: N].argmax(dim=1)
    numbers2 = inputs[:, N: 2 * N].argmax(dim=1)
    numbers = torch.stack([numbers1, numbers2], dim=1)
    return numbers


def calc_random_mae(logs):

    numbers = one_hot_to_numbers(logs.sender_input)
    sums = (torch.sum(numbers, dim=1))

    random_mae = []
    for i in range(10):
        random_sums = torch.randint(0, max(sums), size=(1, len(sums)))
        random_mae.append(torch.mean(torch.abs(random_sums - sums).float()))

    return sum(random_mae) / len(random_mae)


def calc_performance(logs):

    numbers = one_hot_to_numbers(logs.sender_input)
    sums = (torch.sum(numbers, dim=1))

    accuracy = torch.mean((sums == logs.receiver_output.argmax(dim=1)).float())
    mae = torch.mean(torch.abs(logs.receiver_output.argmax(dim=1) - sums).float())

    return accuracy, mae


def calc_mean_bias(logs):

    N = logs.sender_input.shape[1] // 2

    numbers = one_hot_to_numbers(logs.sender_input)

    small_numbers = torch.where(torch.sum(numbers, dim=1) < N)[0]
    large_numbers = torch.where(torch.sum(numbers, dim=1) >= N)[0]
    mean_bias = (torch.mean(logs.receiver_output[small_numbers].argmax(dim=1)
                            - torch.sum(numbers[small_numbers], dim=1).float()),
                 torch.mean(logs.receiver_output[large_numbers].argmax(dim=1)
                            - torch.sum(numbers[large_numbers], dim=1).float()))

    return mean_bias


def joint_entropy(xs, ys):
    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    return calc_entropy(xys)


def information_scores(logs, normalizer="arithmetic"):
    """calculate entropy scores: mutual information (MI), polysemy, and synonymy
    """

    numbers = one_hot_to_numbers(logs.sender_input)
    sums = torch.sum(numbers, dim=1)
    messages = logs.message.argmax(dim=-1)

    H_sums = calc_entropy(sums)
    H_numbers = calc_entropy(numbers)
    H_messages = calc_entropy(messages)

    H_sums_messages = joint_entropy(sums, messages)
    H_numbers_messages = joint_entropy(numbers, messages)

    if normalizer == "arithmetic":
        normalizer_sums_messages = 0.5 * (H_sums + H_messages)
        normalizer_numbers_messages = 0.5 * (H_numbers + H_messages)
    elif normalizer == "joint":
        normalizer_sums_messages = H_sums_messages
        normalizer_numbers_messages = H_numbers_messages
    else:
        raise AttributeError("Unknown normalizer")

    # normalized mutual information
    NMI_sums_messages = (H_sums + H_messages - H_sums_messages) / normalizer_sums_messages
    NMI_numbers_messages = (H_numbers + H_messages - H_numbers_messages) / normalizer_numbers_messages

    # normalized conditional entropy H(input | messages) --> polysemy
    polysemy_numbers = (H_numbers_messages - H_messages) / H_numbers
    polysemy_sums = (H_sums_messages - H_messages) / H_sums

    # normalized conditional entropy H(messages | input) --> synonymy

    synonymy_numbers = (H_numbers_messages - H_numbers) / H_messages
    synonymy_sums = (H_sums_messages - H_sums) / H_messages

    score_dict = {'NMI_sum': NMI_sums_messages,
                  'NMI_summands': NMI_numbers_messages,
                  'polysemy_sum': polysemy_sums,
                  'polysemy_summands': polysemy_numbers,
                  'synonymy_sum': synonymy_sums,
                  'synonymy_summands': synonymy_numbers}

    return score_dict


def synonymy_dict(inputs, messages):

    synonymy = {}

    for i, number in enumerate(inputs):
        key = str(number.numpy())
        if key in synonymy.keys():
            synonymy[key].append(messages[i].item())
        else:
            synonymy[key] = [messages[i].item()]

    for key in synonymy.keys():
        synonymy[key] = set(synonymy[key])

    count_values = [len(val) for val in synonymy.values()]

    return synonymy, np.mean(count_values), np.median(count_values)


def polysemy_dict(inputs, messages):

    polysemy = {}

    for i, m in enumerate(messages):
        key = m.item()
        if key in polysemy.keys():
            polysemy[key].append(str(inputs[i].numpy()))
        else:
            polysemy[key] = [str(inputs[i].numpy())]

    for key in polysemy.keys():
        polysemy[key] = set(polysemy[key])

    count_values = [len(val) for val in polysemy.values()]

    return polysemy, np.mean(count_values), np.median(count_values)




