import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, Dense, MaxPooling2D, Dropout, BatchNormalization, Flatten

from nptrain import is_win

class Game:
    def __init__(self, size=8):
        '''Creates a more optimized Gomoku game to train the A.I.'''

        # The board is where the game is played. 
        # Player 1's stones will be stored on board[y, x, 0]
        # Player 2's stones will be stored on board[y, x, 1]
        self.board = np.zeros((size, size, 2,), dtype=np.float32)

    def is_win(self):
        return is_win(self.board)

    def move(self, y, x, player):
        if(player != 1 and player != 2) or self.board[y, x, 0] or self.board[y, x, 1]:
            raise ValueError
        self.board[y, x, player - 1] = 1.0


    def force_move(self, y, x, player):
        self.board[y, x, player - 1] = 1.0

