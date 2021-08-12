""" plot.py
    Code to plot an example inputs and output from easy-to-hard-data datasets
    Developed as part of the DeepThinking project
    Avi Schwarzschild
    July 2021
"""

import os

import chess.svg
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import torch

from easy_to_hard_data import MazeDataset, ChessPuzzleDataset


def char_from_piece_vec(piece, who_moves):
    chars = "oKQBNRPkqbnrp" if who_moves == 0 else "okqbnrpKQBNRP"
    idx = (piece == 1).nonzero(as_tuple=False)
    if len(idx) > 0:
        idx = idx.item() + 1
    else:
        idx = 0
    return chars[idx]


def chess_board_to_str(board_tensor, who_moves):
    """ function to move from 12x8x8 tensor to FEN representation
    Note: The rows and cols in the tensor correspond to ranks and files
          in the same order, i.e. first row is Rank 1, first col is File A."""

    board_tensor = board_tensor.squeeze()
    board_list = []
    for i in range(7, -1, -1):
        for j in range(8):
            piece = board_tensor[:, i, j]
            board_list.append(char_from_piece_vec(piece, who_moves))
        board_list.append("/")
    board_list = board_list[:-1]
    board_str = []
    o_cnt = 0
    for b in board_list:
        if b != "o" and o_cnt == 0:
            board_str.append(b)
        elif b != "o":
            board_str.append(str(o_cnt))
            board_str.append(b)
            o_cnt = 0
        else:
            o_cnt += 1
    if o_cnt > 0:
        board_str.append(str(o_cnt))
    board_str = "".join(board_str)
    return board_str


def plot_maze(inputs, targets, save_str):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    ax.imshow(inputs.cpu().squeeze().permute(1, 2, 0))

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    ax = axs[1]
    sns.heatmap(targets, ax=ax, cbar=False, linewidths=0, square=True, rasterized=True)

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(save_str, bbox_inches="tight")
    plt.close()


def plot_chess_puzzle(inputs, targets, who_moves, save_str):

    dim = inputs.size(2)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # We rotate targets here since the data comes in oriented with square A1 at
    # the bottom left of the board, so we reorient the targets to match the rotated
    # board if black is to move.
    if who_moves:
        targets = torch.rot90(targets, 2, [0, 1])

    sns.heatmap(targets.reshape(dim, dim), ax=ax, cbar=False, linewidths=1, linecolor='white')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(f"target_{save_str}")
    plt.close()

    board_str = chess_board_to_str(inputs, who_moves)
    print(board_str)
    board = chess.Board(board_str)
    mysvg = chess.svg.board(board, size=350, flipped=bool(who_moves))

    tmp_str = ".easy_to_hard_data_plot_temp.svg"
    with open(tmp_str, "w") as fh:
        fh.write(mysvg)

    drawing = svg2rlg(tmp_str)
    renderPDF.drawToFile(drawing, f"input_{save_str[:-4]}.pdf")
    os.remove(tmp_str)


def main():
    import torchvision.transforms as trans
    t = trans.RandomCrop(32, padding=8)
    mazes = MazeDataset("./data", transform=t)
    inputs, targets = mazes[0]
    plot_maze(inputs, targets, "maze_example_15.png")

    mazes = MazeDataset("./data", train=False, size=33)
    inputs, targets = mazes[0]
    plot_maze(inputs, targets, "maze_example_33.png")

    chess_puzzles = ChessPuzzleDataset("./data", idx_start=0, idx_end=4)
    inputs, targets, who_moves = chess_puzzles[3]
    plot_chess_puzzle(inputs, targets, who_moves, f"chess_example.png")


if __name__ == "__main__":
    main()
