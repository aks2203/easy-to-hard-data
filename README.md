# Easy-To-Hard Datasets
This repository houses Pytorch datasets for our [Easy-To-Hard](http://github.com/aks2203/easy-to-hard) project and a short paper describing these datasets [here](https://arxiv.org/abs/2108.06011).


[![PyPI](https://img.shields.io/pypi/v/easy-to-hard-data)](https://pypi.org/project/easy-to-hard-data/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/aks2203/easy-to-hard-data/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/easy-to-hard-data?period=month&units=international_system&left_color=blue&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/easy-to-hard-data)

## Overview

This package contains datasets desinged for use in studying easy to hard generalization. That is, the training data consists of easy examples, and the testing data has harder ones. The datsets are as follows.

1. Prefix Sums

- Compute the prefix sum modulo two of a binary input string. 
- The length of the string determines the difficulty of the problem.
- We provide 52 different sets (10,000 samples per length) from which to choose one for training data and a longer one for testing.

2. Mazes

- Visually solve a maze where the input is a three channel image, and the output is a binary segmentation mask, which is the same size as the input, separating pixels, with ones at locations that are on the optimal path and zeros elsewhere.
- We provide many size mazes (see below for details).

3. Chess Puzzles
- Choose the best next move given a mid-game chess board.
- The difficulty is determined by the [Lichess](https://lichess.org/training) puzzle rating.
- We sorted the chess puzzles provided by Lichess, and the first 600,000 easiest puzzles make up an easy training set. Testing can be done with any subset of puzzles with higher indices. The default test set uses indices 600,000 to 700,000.

Note that in this repository there are scripts to make data for prefix sums and for mazes and a script to convert Lichess csv data into torch tensors. Also, we include plotting code for mazes and for chess puzzles.

# Installation

This package can be installed with `pip` using the following command:

```pip install easy-to-hard-data```

Then, the datasets can be imported using 

```from easy_to_hard_data import PrefixSumDataset, MazeDataset, ChessPuzzleDataset```

Also, plotting code can be accessed using

```from easy_to_hard_plot import plot_chess_puzzle, plot_maze```

This package can also be installed from source, by cloning the repository as follows.

``` 
git clone https://github.com/aks2203/easy-to-hard-data.git
cd easy-to-hard-data
pip install -e .
```

# Release Notes

Last major release: v1.0.0 (August 2021). The latest version includes major changes to both the raw data files and to the signatures of the dataset class constructors. The old version is still usable, and the automatic downloads will pull the appropriate version of the data.

The changes to the data include new mazes generated with a technique that guarentees unique solutions. Also, the data generation and plotting scripts have been added. Lastly, it is important to note the changes to the constructor signatures, as they have been improved.

# Usage
The intended use for this package is to provide easy-to-use and ready-to-download datasets in PyTorch for those interested in studying generalization from easy to hard problems. Each of the datasets has options detailed below.

## Prefix Sums

<p align='center'>
  <img width='70%' src='https://github.com/aks2203/easy-to-hard-data/blob/main/img/prefix_sum_example.png'/>
</p>

For each sequence length, we provide a set of 10,000 input/output pairs. The `__init__` method has the following signature:

```
PrefixSumDataset(self, root: str, num_bits: int = 32, download: bool = True)
```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `num_bits` arument determines the length of the input sequences, and therefore the difficulty of the problem. The default value is 32, but the avaialable options are 16 through 64 as well as 72, 128, 256, and 512. Finally, the `download` argument sets whether to download the data.

## Mazes

<p align='center'>
  <img width='25%' src='https://github.com/aks2203/easy-to-hard-data/blob/main/img/maze_example_15.png'/>
</p>
<p align='center'>
  <img width='65%' src='https://github.com/aks2203/easy-to-hard-data/blob/main/img/maze_example_33.png'/>
</p>

For sizes in {9, 11, 13, 15, 17} we have 50,000 training examples and 10,000 testing examples. For the larger sizes {19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59}, we provide 1,000 testing mazes each. The mazes shown above are examples of sizes 15 (top) and 33 (bottom). The `__init__` method has the following signature:

```
MazeDataset(self, root: str, train: bool = True, size: int = 9, transform: Optional[Callable] = None, download: bool = True)
```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `train` arument distiguishes between the training and testing sets. The `size` arument sets the size (one of the integers listed above). The `transform` argument allows you to pass in a torchvision transform like random cropping. Finally, the `download` argument sets whether to download the data.

## Chess Puzzles

<p align='center'>
  <img width='40%' src='https://github.com/aks2203/easy-to-hard-data/blob/main/img/chess_input_example.png'/>
  <img width='40%' src='https://github.com/aks2203/easy-to-hard-data/blob/main/img/chess_target_example.png'/>
</p>

We compiled a dataset from Lichess's puzzles database. We provide a set of about 1.5M input/output pairs sorted by dificulty rating. The `__init__` method has the following signature:

```
ChessPuzzleDataset(root: str, train: bool = True, idx_start: int = None, idx_end: int = None, who_moves: bool = True, download: bool = True)
```

The `root` argument must be provided and determines where the data is or to where it will be downloaded if it does not already exist at that location. The `train` arument distiguishes between the training and testing sets. The `idx_start` and `idx_end` aruments are an alternative to `train` and can be used to manually choose the indices in the sorted data to use. The `who_moves` argument returns a boolean, where True indicates that black moves next, and False indicates that white moves next. Finally, the `download` argument sets whether to download the data.

The automatic download will also retrieve a file containing the rating of each chess puzzle. This file is not used by any of the functions/methods in this code, but it is available to be read/used by anyone interested. The indices match the other tensors downloaded. Also, note that if generating the data, [this CSV file](https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/deepthinking_lichess.tar.gz) needs to be downloaded and stored in the same directory as the dataset generation code.

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

## Contributing

If you have improvements or find bugs, feel free to open issues.

## Citing Our Work

If you find this code helpful and use these datasets, please consider citing our work.

The datasets are descried in [Datasets for Studying Generalization from Easy to Hard Examples](https://arxiv.org/abs/2108.06011)

```
@misc{schwarzschild2021datasets,
      title={Datasets for Studying Generalization from Easy to Hard Examples}, 
      author={Avi Schwarzschild and Eitan Borgnia and Arjun Gupta and Arpit Bansal and Zeyad Emam and Furong Huang and Micah Goldblum and Tom Goldstein},
      year={2021},
      eprint={2108.06011},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
