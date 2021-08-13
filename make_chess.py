""" make_chess.py
    For generating chess puzzle dataset for the
    DeepThinking project.
    Eitan Borgnia, Avi Schwarzschild, Zeyad Emam, Arpit Bansal
    August 2021
"""

import datetime

import chess
import pandas as pd
import torch
from tqdm import tqdm


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


def get_board_tensor(board_str, black_moves):
    """ function to move from FEN representation to 12x8x8 tensor
    Note: The rows and cols in the tensor correspond to ranks and files
          in the same order, i.e. first row is Rank 1, first col is File A.
          Also, the color to move next occupies the first 6 channels."""

    p_to_int = {"k": 1, "q": 2, "b": 3, "n": 4, "r": 5, "p": 6,
                "K": 7, "Q": 8, "B": 9, "N": 10, "R": 11, "P": 12}
    new_board = torch.zeros(8, 8)
    rank = 7
    file = 0
    for p in board_str:
        if p == "/":
            rank -= 1
            file = 0
        elif not p.isdigit():
            new_board[rank, file] = (p_to_int[p])
            file += 1
        else:
            new_board[rank, file:file+int(p)] = 0
            file += int(p)

    board_tensor = torch.zeros(12, 8, 8)
    pieces = "kqbnrpKQBNRP" if black_moves else "KQBNRPkqbnrp"
    for p_i, p in enumerate(pieces):
        board_tensor[p_i] = new_board == p_to_int[p]

    return board_tensor


def get_moves_tensor(move):
    file_to_num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    origin_file = file_to_num[move[0]]
    origin_rank = int(move[1]) - 1
    dest_file = file_to_num[move[2]]
    dest_rank = int(move[3]) - 1
    move = torch.zeros(8, 8, dtype=torch.long)
    move[origin_rank, origin_file] = 1
    move[dest_rank, dest_file] = 1
    return move


def generate_tensors(path):

    with open(path, "r") as fh:
        chess_df = pd.read_csv(fh, names=["PuzzleId", "FEN", "Moves", "Rating",
                                          "RatingDeviation", "Popularity", "NbPlays",
                                          "Themes", "GameUrl"])
    chess_df.sort_values("Rating", inplace=True, kind="mergesort")
    num_rows = chess_df.shape[0]
    puzzle_tensor = torch.zeros(num_rows, 12, 8, 8)
    moves_tensor = torch.zeros(num_rows, 8, 8, dtype=torch.long)
    who_moves = torch.zeros(num_rows, 1)
    rating = torch.zeros(num_rows, 1)

    # Iterate through each row of puzzle data
    for i, row_tuple in tqdm(enumerate(chess_df.iterrows())):
        row = row_tuple[1]
        move = chess.Move.from_uci(row["Moves"].split(" ")[0])
        board = chess.Board(row["FEN"])
        board.push(move)
        new_fen = board.fen()

        black_moves = {"w": 0, "b": 1}[new_fen.split(" ")[1]]
        who_moves[i] = black_moves
        rating[i] = row["Rating"]

        board_str = new_fen.split(" ")[0]
        puzzle_tensor[i] = get_board_tensor(board_str, black_moves)
        moves_tensor[i] = get_moves_tensor(row["Moves"].split(" ")[1])

        if i % 10000 == 0:
            print(f"iteration {i}/{chess_df.shape[0]}, {datetime.datetime.now().strftime('%H:%M:%S')}")

    return puzzle_tensor, moves_tensor, who_moves, rating


def main():
    data, targets, who_moves, rating = generate_tensors("deepthinking_lichess.csv")
    torch.save(targets, "chess_data/targets.pth")
    torch.save(data, "chess_data/data.pth")
    torch.save(who_moves, "chess_data/who_moves.pth")
    torch.save(rating, "chess_data/rating.pth")


if __name__ == "__main__":
    main()
