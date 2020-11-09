from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, BatchNormalization, Flatten

from nptrain import is_win

class Game:
    def __init__(self, board=np.zeros((8, 8, 2,), dtype=np.float32)):
        '''Creates a more optimized Gomoku game to train the A.I.'''

        # The board is where the game is played. 
        # Player 1's stones will be stored on board[y, x, 0]
        # Player 2's stones will be stored on board[y, x, 1]
        self.board = board

    def is_win(self):
        return is_win(self.board)


    def move(self, y, x, player):
        '''Play a move for the player player and returns 1 if the move fails.'''
        if(player != 1 and player != 2) or self.board[y, x, 0] or self.board[y, x, 1]:
            return 1
        self.board[y, x, player - 1] = 1.0


    def force_move(self, y, x, player):
        '''Force a move. Sets the board slot given [y, x, player - 1] to 1.0'''
        self.board[y, x, player - 1] = 1.0


class MCTS:
    def __init__(self, board):
        self.sims = []
        for _ in range(400):
            self.sims.append(Game())
        