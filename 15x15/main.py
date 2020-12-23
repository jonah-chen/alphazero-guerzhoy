"""This file is the main file used to train the neural network model.
Author(s): Muhammad Ahsan Kaleem, Jonah Chen
"""
import numpy as np

from play import generate_data, eval_model
from ai import train_model
from copy import copy
from tensorflow.keras.models import load_model


[best_model_num, num] = np.load("config.npy")

model = load_model(f'models/{num}.h5')
best_model = load_model(f'models/{best_model_num}.h5')


while 1:
    num += 1
    generate_data(num, best_model, games=128, search_iter=1024)
    train_model(model, num=num, log_name=num, epochs=1)
    model.save(f'models/{num}.h5')

    score, record, black_games, white_games = eval_model(model, best_model, games=50, search_iter=1024)

    # Saves the games. Black first, white second.
    np.save(f'games/{num}v{best_model_num}', black_games)
    np.save(f'games/{best_model_num}v{num}', white_games)

    del black_games, white_games

    print(score)
    if score >= 55:
        best_model = copy(model)
        best_model_num = num
    
    np.save("config", np.array([best_model_num, num], dtype=int))

    del score, record
