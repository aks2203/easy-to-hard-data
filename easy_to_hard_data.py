""" easy_to_hard_data.py
    Python package with datasets for studying generalization from
        easy training data to hard test examples.
    Developed as part of easy-to-hard (github.com/aks2203/easy-to-hard).
    Avi Schwarzschild
    June 2021
"""

import errno
import os
import os.path
import tarfile
import urllib.request as ur
from typing import Optional, Callable

import numpy as np
import torch
from tqdm import tqdm

GBFACTOR = float(1 << 30)


def extract_zip(path, folder):
    file = tarfile.open(path)
    file.extractall(folder)
    file.close


def download_url(url, folder):
    filename = url.rpartition('/')[2]
    path = os.path.join(folder, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('Using existing file', filename)
        return path
    print('Downloading', url)
    makedirs(folder)
    # track downloads
    ur.urlopen(f"http://avi.koplon.com/hit_counter.py?next={url}")
    data = ur.urlopen(url)
    size = int(data.info()["Content-Length"])
    chunk_size = 1024*1024
    num_iter = int(size/chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    except:
        if os.path.exists(path):
             os.remove(path)
        raise RuntimeError('Stopped downloading due to interruption.')

    return path


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


class ChessPuzzleDataset(torch.utils.data.Dataset):
    base_folder = "chess_data"
    url = "https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/chess_data.tar.gz"

    def __init__(self, root: str,
                 train: bool = True,
                 idx_start: int = None,
                 idx_end: int = None,
                 who_moves: bool = True,
                 download: bool = True):

        self.root = root
        self.ret_who_moves = who_moves

        if download:
            self.download()

        self.train = train
        if idx_start is None or idx_end is None:
            if train:
                print("Training data using pre-set indices [0, 600000).")
                idx_start = 0
                idx_end = 600000
            else:
                print("Testing data using pre-set indices [600000, 700000).")
                idx_start = 600000
                idx_end = 700000
        else:
            print(f"Custom data range using indices [{idx_start}, {idx_end}].")

        inputs_path = os.path.join(root, self.base_folder, "data.pth")
        solutions_path = os.path.join(root, self.base_folder, "targets.pth")
        who_moves = os.path.join(root, self.base_folder, "who_moves.pth")

        self.puzzles = torch.load(inputs_path)[idx_start:idx_end]
        self.targets = torch.load(solutions_path)[idx_start:idx_end].flip(1)
        self.who_moves = torch.load(who_moves)[idx_start:idx_end]

    def __getitem__(self, index):
        if self.ret_who_moves:
            return self.puzzles[index], self.targets[index], self.who_moves[index]
        else:
            return self.puzzles[index], self.targets[index]

    def __len__(self):
        return self.puzzles.size(0)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.base_folder)
        if not os.path.exists(fpath):
            return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)


class MazeDataset(torch.utils.data.Dataset):
    """This is a dataset class for mazes.
    padding and cropping is done correctly within this class for small and large mazes.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 size: int = 9,
                 transform: Optional[Callable] = None,
                 download: bool = True):

        self.root = root
        self.train = train
        self.size = size
        self.transform = transform

        self.folder_name = f"maze_data_{'train' if self.train else 'test'}_{size}"
        url = f"https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/" \
              f"{self.folder_name}.tar.gz"

        if download:
            self.download(url)

        print(f"Loading mazes of size {size} x {size}.")

        inputs_path = os.path.join(root, self.folder_name, "inputs.npy")
        solutions_path = os.path.join(root, self.folder_name, "solutions.npy")
        inputs_np = np.load(inputs_path)
        targets_np = np.load(solutions_path)

        self.inputs = torch.from_numpy(inputs_np).float()
        self.targets = torch.from_numpy(targets_np).long()

    def __getitem__(self, index):
        img, target = self.inputs[index], self.targets[index]

        if self.transform is not None:
            stacked = torch.cat([img, target.unsqueeze(0)], dim=0)
            stacked = self.transform(stacked)
            img = stacked[:3].float()
            target = stacked[3].long()

        return img, target

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.folder_name)
        if not os.path.exists(fpath):
            return False
        return True

    def download(self, url) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        path = download_url(url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)


class PrefixSumDataset(torch.utils.data.Dataset):
    base_folder = "prefix_sums_data"
    url = "https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/prefix_sums_data.tar.gz"
    lengths = list(range(16, 65)) + [72] + [128] + [256] + [512]
    download_list = [f"{l}_data.pth" for l in lengths] + [f"{l}_targets.pth" for l in lengths]

    def __init__(self, root: str, num_bits: int = 32, download: bool = True):

        self.root = root

        if download:
            self.download()

        print(f"Loading data with {num_bits} bits.")

        inputs_path = os.path.join(root, self.base_folder, f"{num_bits}_data.pth")
        targets_path = os.path.join(root, self.base_folder, f"{num_bits}_targets.pth")
        self.inputs = torch.load(inputs_path).unsqueeze(1) - 0.5
        self.targets = torch.load(targets_path).long()

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.download_list:
            fpath = os.path.join(root, self.base_folder, fentry)
            if not os.path.exists(fpath):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)


if __name__ == "__main__":
    md = MazeDataset("./data")
    cd = ChessPuzzleDataset("./data")
    psd = PrefixSumDataset("./data")
    print("All datasets downloaded.")
