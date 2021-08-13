""" make_prefixsums.py
    For generating prefix sums dataset for the
    DeepThinking project.
    Avi Schwarzschild and Eitan Borgnia
    July 2021
"""

import collections as col

import torch


def binary(x, bits):
    mask = 2**torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).long()


def get_target(inputs):
    cumsum = torch.cumsum(inputs, 0)
    targets = cumsum % 2
    return targets


if __name__ == "__main__":

    # Change this variable, digits, to make datasets with different numbers of binary digits
    for digits in list(range(16, 65)) + [72, 128, 256, 512]:
        inputs = torch.zeros(10000, digits)
        targets = torch.zeros(10000, digits)
        if digits < 24:
            rand_tensor = torch.arange(2 ** digits)[torch.randperm(2 ** digits)[:10000]]
            rand_tensor = torch.stack([binary(r, digits) for r in rand_tensor])
        else:
            rand_tensor = torch.rand(10000, digits) >= 0.5

        for i in range(10000):
            target = get_target(rand_tensor[i])
            inputs[i] = rand_tensor[i]
            targets[i] = target

        torch.save(inputs, f"{digits}_data.pth")
        torch.save(targets, f"{digits}_targets.pth")

        # Check for repeats
        inputs = inputs.numpy()
        t_dict = {}
        t_dict = col.defaultdict(lambda: 0) # t_dict = {*:0}
        for t in inputs:
            t_dict[t.tobytes()] += 1        # t_dict[input] += 1

        repeats = 0
        for i in inputs:
            if t_dict[i.tobytes()] > 1:
                repeats += 1

        print(f"There are {repeats} repeats in the dataset.")
