import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, Flatten, ReLU, Add

from game import Game

LOC = "saved_models"

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
    policy = Dense(64, activation='softmax')(policy)

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
    model = build_model()
    model.summary()
    model.save(LOC)
    
    # import time

    # G = Game()
    # i = 0
    # while G.is_win() == 0:
    #     y = int(input("input y\n"))
    #     x = int(input("input x\n"))
    #     if(G.move(y, x, i % 2 + 1) == 1):
    #         print("Illegal move")
    #     else:
    #         i += 1
    #         start = time.perf_counter()
    #         output = model.predict(np.array([G.black for _ in range(1024)]))
    #         end = time.perf_counter()
    #         print(G)
    #         print(f"Time Taken: {1000*(end-start):.1f}ms\n")