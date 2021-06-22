# Easy-To-Hard Datasets
Pytorch datasets for our [Easy-To-Hard](http://github.com/aks2203/easy-to-hard) project.


[![PyPI](https://img.shields.io/pypi/v/easy-to-hard-data)](https://pypi.org/project/easy-to-hard-data/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snap-stanford/ogb/blob/master/LICENSE)

## Overview

This package contains datasets desinged for use in studying easy to hard generalization. That is, the training data constits of easy examples, and the testing data has harder ones. The datsets are as follows.

1. Prefix Sums

- Compute the prefix sum modulo two of a binary input string. 
- The length of the string determines the difficulty of the problem.
- We provide 53 different sets (10,000 samples per length) from which to choose one for training data and a longer one for testing.

2. Mazes

- Visually solve a maze where the input is a 32 x 32 three channel image, and the output is a binary segmentation mask separating pixels that is the same size as the input with ones at locations that are on the optimal path and zeros elsewhere.
- We provide small and large mazes, which are easy and hard, respectively. 

3. Chess Puzzles
- Choose the best next move given a mid-game chess board.
- The difficulty is determined by the [Lichess](https://lichess.org/training) puzzle rating.
- We sorted the chess puzzles provided by Lichess, and the first 600,000 easiest puzzles make up an easy training set. Testing can be done with any subset of puzzles with higher indices. The default test set uses indices 600,000 to 700,000.

# Installation

This package can be installed with `pip` using the following command:

```pip install easy-to-hard-data```

Then, the datasets can be imported using 

```from easy_to_hard_data import PrefixSumDataset, MazeDataset, ChessPuzzleDataset```

This package can also be installed from source, by cloning the repository as follows.

``` 
git clone https://github.com/aks2203/easy-to-hard-data.git
cd easy-to-hard-data
pip install -e .
```

# Usage
The intended use for this package is to provide easy to use and ready to download datasets in PyTorch for those interested in studying generalization from easy to hard problems. Each of the datasets has options detailed below.

## Prefix Sums

<p align='center'>
  <img width='70%' src='https://aks2203.github.io/easy-to-hard-data/prefix_sum_example.png'/>
</p>

For each sequence length, we provide a set of 10,000 input/output pairs. The `__init__` method has the following signature:

```
PrefixSumDataset(self, root: str, num_bits: int = 32, download: bool = True)
```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `num_bits` arument determines the length of the input sequences, and therefore the difficulty of the problem. The default value is 32, but the avaialable options are 16 through 64 as well as 72 and 128. Finally, the `download` argument sets whether to download the data.

## Mazes

<p align='center'>
  <img width='38%' src='https://aks2203.github.io/easy-to-hard-data/mazes_example_input.png'/>
  <img width='40%' src='https://aks2203.github.io/easy-to-hard-data/mazes_example_target.png'/>
</p>

For each size (small and large), we provide a set of input/output pairs divided into training and testing sets with 50,000 and 10,000 elements, respectively. The `__init__` method has the following signature:

```
MazeDataset(self, root: str, train: bool = True, small: bool = True, download: bool = True)
```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `train` arument distiguishes between the training and testing sets. The `small` arument sets the size (True for small, False for large). Finally, the `download` argument sets whether to download the data.

## Chess Puzzles

<p align='center'>
  <img width='40%' src='https://aks2203.github.io/easy-to-hard-data/chess_input_example.png'/>
  <img width='40%' src='https://aks2203.github.io/easy-to-hard-data/chess_target_example.png'/>
</p>

We compiled a dataset from Lichess's puzzles database. We provide a set of about 1.5M input/output pairs sorted by dificulty rating. The `__init__` method has the following signature:

```
ChessPuzzleDataset(root: str, train: bool = True, idx_start: int = None, idx_end: int = None, download: bool = True)
```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `train` arument distiguishes between the training and testing sets. The `idx_start` and `idx_end` aruments are an alternative to `train` and can be used to manually choose the indices in the sorted data to use. Finally, the `download` argument sets whether to download the data.

## Example

To make two prefix-sum dataloaders, one with training (32 bits) and one with testing (40 bits) data, we provide the following example.

```
from easy_to_hard_data import PrefixSumDataset
import torch.utils.data as data

train_data = PrefixSumDataset("./data", num_bits=32, download=True)
test_data = PrefixSumDataset("./data", num_bits=40, download=True)

trainloader = data.DataLoader(train_data, batch_size=200, shuffle=True)
testloader = data.DataLoader(test_data, batch_size=200, shuffle=False)
```

## Cite our work

If you find this code helpful adn use these datasets, please consider citing our work.

```
@misc{schwarzschild2021learn,
      title={Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks}, 
      author={Avi Schwarzschild and Eitan Borgnia and Arjun Gupta and Furong Huang and Uzi Vishkin and Micah Goldblum and Tom Goldstein},
      year={2021},
      eprint={2106.04537},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
