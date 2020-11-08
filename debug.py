'''Debugger and Helper functions for the Project'''

import numpy as np
# from gomoku import print_board
import trainGomoku as gm

def random_board(shape, bad=False):
    board = np.random.randint(-1,2,size=shape)
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

if __name__=='__main__':
    gm.init()
    gm.print_board()
    coord = input("Give the y value and the x value as, for example, if y = 5, x = 5, input the following: 55")
    y_val = int(coord[0])
    x_val = int(coord[1])
    gm.move(y_val, x_val, 1)
    gm.print_board()
    coord = input("Give the y value and the x value as, for example, if y = 5 , x = 5, input the following: 55")
    x_val - int(coord[0])
    y_val = int(coord[1])
    gm.move(y_val, x_val, 2)
    gm.print_board()

