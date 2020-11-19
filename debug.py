"""Debugger and Helper functions for the Project"""

import numpy as np
# from gomoku import print_board
import trainGomoku as gm
from game import print_board
from nptrain import is_win


def random_board(shape, bad=False):
    board = np.random.randint(-1, 2, size=shape)
    if bad:
        return convert_good_to_bad_board(board)
    return board


def convert_good_to_bad_board(good_board):
    bad_board = good_board.tolist()
    for i in range(len(bad_board)):
        for j in range(len(bad_board[0])):
            if bad_board[i][j] == 0:
                bad_board[i][j] = ' '
            elif bad_board[i][j] == 1:
                bad_board[i][j] = 'b'
            elif bad_board[i][j] == -1:
                bad_board[i][j] = 'w'
    return bad_board


def convert_to_one_hot(bad):
    arr = np.zeros((8, 8, 2,), dtype='float32')
    for i in range(8):
        for j in range(8):
            if bad[i][j] == 'b':
                arr[i, j, 0] = 1.0
            elif bad[i][j] == 'w':
                arr[i, j, 1] = 1.0
    return arr


if __name__ == '__main__':
    prob = [['', ' ', ' ', 'b', ' ', ' ', '', 'w'],
            ['b', '', 'b', 'w', 'w', '', ' ', 'w'],
            ['b', '', 'w', 'b', 'b', 'w', ' ', 'b'],
            [' ', '', 'w', '', 'b', ' ', 'b', ''],
            ['w', 'w', 'w', 'b', 'w', ' ', 'b', 'w'],
            ['w', '', 'b', 'w', 'b', 'w', '', ''],
            ['w', '', 'b', 'b', '', '', 'w', ' '],
            ['b', 'w', 'w', ' ', 'b', '', 'b', ' ']]
    prob = convert_to_one_hot(prob)
    print_board(prob)
    print(is_win(prob))
    # gm.init()
    # gm.print_board()
    # turns = 0
    # while(gm.is_win() == 0):
    #     y = int(input("\nyval"))
    #     x = int(input("\nxval"))
    #     if turns % 2 == 0:
    #         if gm.move(y,x,1) == 0:
    #             turns += 1
    #             gm.print_board()
    #     else:
    #         if gm.move(y,x,2) == 0:
    #             turns += 1
    #             gm.print_board()
    # pie = np.load('selfplay_data/0000/pie.npy')
    # z = np.load('selfplay_data/0000/z.npy')
    # s = np.load('selfplay_data/0000/s.npy')
    #
    # print(f'{pie.shape} {z.shape} {s.shape}')
    #
    # for i in range(100):
    #     print(pie[i])
    #     print_board(s[i])
