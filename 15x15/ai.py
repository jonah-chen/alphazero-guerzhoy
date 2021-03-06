"""Constructs the Neural Network described in D. Silver et. al. Mastering the game of Go without human knowldege, Nature 550, 354-359(2017) doi:10.1038/nature24270.

Author(s): Jonah Chen, Muhammad Ahsan Kaleem
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, Flatten, ReLU, Add
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from random import randint

from game import Game

L2_VAL = 1e-4


def residual_block(x):
    """Build the residual block described in the paper.
    """
    y = Conv2D(256, (3, 3), strides=1, padding='same',
               kernel_regularizer=l2(l=L2_VAL))(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(256, (3, 3), strides=1, padding='same',
               kernel_regularizer=l2(l=L2_VAL))(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = ReLU()(out)
    return out


def build_model(input_shape=(15, 15, 2)):
    """Build the ResNet with the additional features described in the paper.
    """
    # Process input
    inputs = Input(shape=input_shape)

    # Create convolutional block
    t = Conv2D(256, (3, 3), strides=1, padding='same',
               kernel_regularizer=l2(l=L2_VAL))(inputs)
    t = BatchNormalization()(t)
    t = ReLU()(t)

    # Create 19 residual blocks
    for _ in range(19):
        t = residual_block(t)

    # Create policy head
    policy = Conv2D(2, (1, 1), strides=1, kernel_regularizer=l2(l=L2_VAL))(t)
    policy = BatchNormalization()(policy)
    policy = ReLU()(policy)
    policy = Flatten()(policy)
    policy = Dense(15*15, activation='softmax', kernel_regularizer=l2(
        l=L2_VAL), name="policy")(policy)

    # Create value head
    value = Conv2D(1, (1, 1), strides=1, kernel_regularizer=l2(l=L2_VAL))(t)
    value = BatchNormalization()(value)
    value = ReLU()(value)
    value = Flatten()(value)
    value = Dense(256, activation='relu',
                  kernel_regularizer=l2(l=L2_VAL))(value)

    value = Dense(1, activation='tanh', kernel_regularizer=l2(
        l=L2_VAL), name="value")(value)
    # Build model
    model = Model(inputs=inputs, outputs=[policy, value])

    return model


def compile_new_model(loc, lr=1e-2, model=None):
    """Compiles a model with the proper parameters. 
    If no model is given, a new model will be created. 
    The model is saved to the location loc.
    """
    test_vector = np.random.randint(0, 2, size=(3, 15, 15, 2,))
    if model is None:
        model = build_model()
    print(model.predict(test_vector))
    model.compile(loss=[categorical_crossentropy, mean_squared_error], optimizer=SGD(
        lr=lr, momentum=0.9), metrics=['accuracy', 'mean_absolute_error'])
    model.summary()
    print(model.predict(test_vector))
    model.save(loc)


def train_model(model, num=None, s=None, pie=None, z=None, log_name=None, epochs=20, batch_size=32, transformations=True, max_history=20):
    """Loads the training data to train the model and trains it with the given parameters.
    """
    if num is not None:
        pie = np.load(f'selfplay_data/{num}/pie.npy')
        z = np.load(f'selfplay_data/{num}/z.npy')
        s = np.load(f'selfplay_data/{num}/s.npy')
        with ProcessPoolExecutor() as executor:
            for i in range(max(1, num-max_history), num):
                pie = executor.submit(np.append, pie, np.load(
                    f'selfplay_data/{i}/pie.npy'), 0).result()
                z = executor.submit(np.append, z, np.load(
                    f'selfplay_data/{i}/z.npy'), 0).result()
                s = executor.submit(np.append, s, np.load(
                    f'selfplay_data/{i}/s.npy'), 0).result()
    elif s is None or pie is None or z is None:
        raise NotImplementedError(message="Did not recieve training data")

    if transformations:
        for i in range(s.shape[0]):
            rotation, reflection = randint(0, 3), randint(0, 1)
            if rotation:
                s[i] = np.rot90(s[i], k=rotation, axes=(0, 1))
                pie[i] = np.rot90(pie[i].reshape(15,15,), k=rotation, axes=(0, 1)).reshape(225)
            if reflection:
                s[i] = np.flip(s[i], axis=0)
                pie[i] = np.flip(pie[i].reshape(15,15), axis=0).reshape(225)


    if log_name is None:
        model.fit(x=s, y=[pie, z], batch_size=batch_size,
                  epochs=epochs, shuffle=True, use_multiprocessing=True)
    else:
        # Create the callbacks
        model_checkpoint = ModelCheckpoint(
            'checkpoint', monitor='loss', save_best_only=True)
        tensorboard = TensorBoard(
            log_dir=f'LOGS/{log_name}', histogram_freq=1, write_images=True)

        model.fit(x=s, y=[pie, z],
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  use_multiprocessing=True,
                  callbacks=[tensorboard, model_checkpoint])

        # Load the weights from the checkpoint
        model = tf.keras.models.load_model('checkpoint')


def test_model(model, num):
    """Evaluates the model on a set of data to find loss and other metrics."""
    pie = np.load(f'selfplay_data/{num}/pie.npy')
    z = np.load(f'selfplay_data/{num}/z.npy')
    s = np.load(f'selfplay_data/{num}/s.npy')
    model.evaluate(x=s, y=[pie, z], batch_size=32, use_multiprocessing=True)


if __name__ == '__main__':
    compile_new_model("models/0.h5")
