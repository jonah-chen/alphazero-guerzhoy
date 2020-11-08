import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, Dense, MaxPooling2D, Dropout, BatchNormalization, Flatten

# Some basic metrics to assist the first training attempts


class Game:
    def __init__(self, size):
        '''Creates a more optimized Gomoku game to train the A.I.'''
        self.board = np.zeros((size, size,))
        self.board_size = size
        self.is_win = is_win
