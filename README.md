# Easy-To-Hard Datasets
Pytorch datasets for our [Easy-To-Hard](http://github.com/aks2203/easy-to-hard) project. 

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
  <img width='70%' src='https://github.com/aks2203/aks2203.github.io/blob/master/images/prefix_sum_example.png'/>
</p>

For each sequence length, we provide a set of 10,000 input/output pairs. The `__init__` method has the following signature:
```def __init__(self, root: str, num_bits: int = 32, download: bool = True):```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `num_bits` arument determines the length of the input sequences, and therefore the difficulty of the problem. The default value is 32, but the avaialable options are [12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 72, 128]. Finally, the `download` argument sets whether to download the data.