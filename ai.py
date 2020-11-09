from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, Flatten, ReLU, Add

from nptrain import is_win

LOC = "saved_models"

class Game:
    def __init__(self, black=np.zeros((8, 8, 2,), dtype=np.float32), white=np.zeros((8, 8, 2,), dtype=np.float32)):
        '''Creates a more optimized Gomoku game to train the A.I.'''

        # The stones will be stored in a way to feed into the neural network
        # One's own stones are stored on [y, x, 0] 
        # and the opponent's stones are stored on [y, x, 1]

        # The black is where the game is played from the black's perspective 
        # Player 1's stones are stored on black[y, x, 0]
        # Player 2's stones are stored on black[y, x, 1]
        self.black = black

        # The white is where the game is played from the white's perspective 
        # Player 2's stones are stored on white[y, x, 0]
        # Player 1's stones are stored on white[y, x, 1]
        self.white = white

    def is_win(self):
        '''Return the game state as the index in the array ["White won", "Black won", "Draw", "Continue Playing"]'''
        return is_win(self.black)


    def move(self, y, x, player):
        '''Play a move for the player player and returns 1 if the move fails.'''
        if(player != 1 and player != 2) or self.black[y, x, 0] or self.black[y, x, 1]:
            return 1
        self.black[y, x, player - 1] = 1.0
        self.white[y, x, player - 2] = 1.0


    def force_move(self, y, x, player):
        '''Force a move. Sets the black slot given [y, x, player - 1] to 1.0'''
        self.black[y, x, player - 1] = 1.0
        self.white[y, x, player - 2] = 1.0


def residual_block(x):
    '''Build the residual block described in the paper.'''
    y = Conv2D(256, (3, 3), strides=1, padding='same')(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(256, (3,3), strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = ReLU()(out)
    return out
    

def build_model(input_shape=(8,8,2)):
    '''Build the ResNet with the additional features described in the paper.'''
    # Process input
    inputs = Input(shape=input_shape)

    # Create convolutional block
    t = Conv2D(256, (3, 3), strides=1, padding='same')(inputs)
    t = BatchNormalization()(t)
    t = ReLU()(t)

    # Create 19 residual blocks
    for _ in range(19):
        t = residual_block(t)
    
    # Create policy head
    policy = Conv2D(2, (1, 1), strides=1)(t)
    policy = BatchNormalization()(policy)
    policy = ReLU()(policy)
    policy = Flatten()(policy)
    policy = Dense(64, activation='linear')(policy)

    # Create value head
    value = Conv2D(1, (1, 1), strides=1)(t)
    value = BatchNormalization()(value)
    value = ReLU()(value)
    value = Flatten()(value)
    value = Dense(256, activation='relu')(value)
    value = Dense(1, activation='tanh')(value)

    # Build model
    model = Model(inputs=inputs, outputs=[policy, value])

    return model



if __name__ == '__main__':
    model = tf.keras.models.load_model(LOC)
    input_vector = np.ones((3,8,8,2,), dtype='float32')
    output = model.predict(input_vector)
    print(output[1])
    print(output[0])