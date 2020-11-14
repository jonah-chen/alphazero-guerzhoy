import numpy as np

from play import generate_data, eval_model
from ai import train_model
from copy import copy
from tensorflow.keras.models import load_model


model = load_model('models/4')
best_model_num = 4
best_model = load_model('models/4')

for num in range(5, 25):
    generate_data(num, best_model) 
    train_model(model, num=num, log_name=num, epochs=20)
    model.save(f'models/{num}')
    
    score, record, black_games, white_games = eval_model(model, best_model)

    # Saves the games. Black first, white second.
    np.save(f'games/{num}v{best_model_num}', black_games)
    np.save(f'games/{best_model_num}v{num}', white_games)

    del black_games, white_games

    print(score)
    if score >= 55:
        best_model = copy(model)
        best_model_num = num
        