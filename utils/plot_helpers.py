import torch
from utils.analysis_helpers import one_hot_to_numbers
from matplotlib import pyplot as plt
import pandas as pd


def plot_accuracy_versus_frequency(logs):

    N = logs.sender_input.shape[1] // 2

    numbers = one_hot_to_numbers(logs.sender_input)
    receiver_output = torch.argmax(logs.receiver_output, dim=1)

    results = {'sum': [], 'frequency': [], 'accuracy': []}
    input_sums = torch.sum(numbers, dim=1)
    for sum_i in range(2*N-1):
        indices = torch.where(input_sums == sum_i)[0]
        if len(indices) > 0:
            results['sum'].append(sum_i)
            results['frequency'].append(len(indices) / len(input_sums))
            accuracy = torch.mean((input_sums[indices] == receiver_output[indices]).float())
            results['accuracy'].append(accuracy)
    plt.scatter(results['frequency'], results['accuracy'])
    plt.xlabel('frequency')
    plt.ylabel('accuracy')
    return results


def plot_symbol_per_input(logs):

    numbers = one_hot_to_numbers(logs.sender_input)
    messages = logs.message.argmax(dim=-1)
    N = logs.sender_input.shape[1] // 2

    x = torch.tensor(numbers[:, 0])
    y = torch.tensor(numbers[:, 1])

    n = len(x)
    image = torch.zeros((N, N))

    for i in range(n):
        image[x[i], N - 1 - y[i]] = messages[i]

    plt.imshow(image, cmap='gray')
    plt.xticks(torch.arange(0, N, 5))
    plt.yticks(ticks=torch.arange(0, N, 5)[::-1], labels=torch.arange(0, N, 5))
    plt.show()


def symbols_to_plot_labels(synonymy_dict, sum_size=41):

    str_values = []
    for s in range(sum_size):
        vals = synonymy_dict[str(s)]
        string = ''
        for elem in vals:
            string += str(elem) + '\n'
        str_values.append(string)
    return str_values


def sums_to_plot_labels(polysemy_dict, V_size=82):

    str_values = []
    symbols = []
    for s in range(V_size):
        try:
            vals = polysemy_dict[str(s)]
            string = ''
            for elem in vals:
                string += str(elem) + '\n'
            str_values.append(string)
            symbols.append(s)
        except KeyError:
            continue
    return str_values, symbols


def get_synonymy_polysemy_dfs(synonymy, polysemy):

    x = [int(k) for k in synonymy.keys()]
    y = [len(values) for values in synonymy.values()]
    df_synonymy = pd.DataFrame({'x': x, 'y': y})

    x = [int(k) for k in polysemy.keys()]
    y = [len(values) for values in polysemy.values()]
    df_polysemy = pd.DataFrame({'x': x, 'y': y})

    return df_synonymy, df_polysemy


