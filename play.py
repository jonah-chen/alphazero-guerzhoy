from time import perf_counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf

from mcts import optimized_search
from game import move_on_board
from nptrain import *


def self_play(model, games=128, game_iter=64, search_iter=512):
    boards = np.zeros((games,8,8,2,), dtype="float32")
    players = [1]*games
    inputs = None

    s = []
    pie = []
    z = []

    # These are the parameters to train the network to gained by MCTS process
    # The elements are accessed as game_boards[game#][turn#]
    game_boards = [[] for _ in range(games)]
    mcts_policies = [[] for _ in range(games)]

    

    for turns in range(game_iter):
        print(f"------------------------------------------------------------\nTurn {turns+1} of {game_iter}. Cumulated: {int(perf_counter() - true_start)}s")
        if len(game_boards) == 0:
            return s, pie, z
        results = optimized_search(model, boards, players, roots=inputs, it=search_iter)
        inputs = []
        games_ended = 0
        for j in range(len(results)):
            i = j - games_ended

            # Save the results of the MCTS to train NN
            act, dist = results[i].play()
            game_boards[i].append(deepcopy(boards[i] if players[i] == 1 else np.flip(boards[i], axis=2)))
            mcts_policies[i].append(dist)

            # Make Move
            move_on_board(boards[i], act, player=players[i])
            
            # When game ends, save the data of the game.
            state = is_win(boards[i])
            if state:
                s.append(game_boards.pop(i))
                pie.append(mcts_policies.pop(i))
                if state == 1:
                    z.append([1 - 2 * (k % 2) for k in range(turns+1)])
                elif state == 2:
                    z.append([2 * (k % 2) - 1 for k in range(turns+1)])
                elif state == 3:
                    z.append([0]*(turns+1))
                boards = np.delete(boards, i, axis=0)
                players.pop()
                
                games_ended += 1
            else:
                # When game doesn't end. Player changes and the new state is appended to be evaluated on the next tern.
                inputs.append(results[i].children[act])
                players[i] = players[i] % 2 + 1

    return s, pie, z


def digest(list_of_list):
    temp = []
    for x1 in list_of_list:
        for x2 in x1:
            temp.append(x2)
    return np.array(temp)


def ai_v_ai(black, white, games=64, game_iter=64, search_iter=512, tau=0):
    """Plays the AI black against white. Return the score of black (between 0 and 100, higher is better), the list of list of games played as moves (0-63) in the order they are played, and the record as a tuple (losses, draws, wins). Black will start with the black stones in every game"""
    
    # Creates the boards.
    boards = np.zeros((games,8,8,2,), dtype="float32")
    players = [1]*games
    inputs = None

    # Create the statistics.
    wins, losses, draws = 0, 0, 0

    # Creates the arrays of the moves being made.
    temp_games = [[] for _ in range(games)]
    save_games = []

    for turns in range(game_iter):
        print(f"------------------------------------------------------------\nTurn {turns+1} of {game_iter}. w/d/l={wins}/{draws}/{losses}")
        # Return when all games end
        if len(temp_games) == 0:
            return round((100*wins+50*draws)/games), save_games, [losses, draws, wins]

        # Execute the MCTS
        results = optimized_search(white if turns % 2 else black, boards, players, roots=inputs, it=search_iter)

        inputs = []
        games_ended = 0

        for j in range(len(results)):
            i = j - games_ended

            # Generate and make the move
            act, _ = results[i].play(tau=tau)
            move_on_board(boards[i], act, player=players[i])
            temp_games[i].append(act)

            # When game ends, save the data of the game.
            state = is_win(boards[i])
            if state:
                save_games.append(temp_games.pop(i))
                if state == 1:
                    wins += 1
                elif state == 2:
                    losses += 1
                elif state == 3:
                    draws += 1
                boards = np.delete(boards, i, axis=0)
                players.pop()
                
                games_ended += 1
            else:
                # When game doesn't end. Player changes and the new state is appended to be evaluated on the next tern.
                inputs.append(results[i].children[act])
                players[i] = players[i] % 2 + 1
        
    return round((100*wins+50*draws)/games), save_games, [losses, draws, wins]


def generate_data(num, model, games=128):
    global true_start
    true_start = perf_counter()

    # Check if a directory exists
    np.save(f'selfplay_data/{num}/_test', np.zeros(1,))
    print("The given directory is valid.")

    s, pie, z = self_play(model, games=games)
    
    start = perf_counter()
    with ProcessPoolExecutor() as executor:
        pie_f = executor.submit(digest, pie)
        z_f = executor.submit(digest, z)
        s_f = executor.submit(digest, s)

        pie = pie_f.result()
        z = z_f.result()
        s = s_f.result()
    end = perf_counter()
    print(end-start)
    
    np.save(f'selfplay_data/{num}/pie', pie)
    np.save(f'selfplay_data/{num}/z', z)
    np.save(f'selfplay_data/{num}/s', s)


def eval_model(new_model, old_model, games=128):
    """Play games games with equal chance each model gets white and black and return 
    the score the new_model achieved(0-100), 
    the record [losses, draws, wins], 
    the games played with black, 
    the games played with white 
    as a tuple in order."""
    _, games1, record1 = ai_v_ai(new_model, old_model, games=games//2)
    _, games2, record2 = ai_v_ai(old_model, new_model, games=games//2)
    return round(((record1[2]+record2[0])*100 + (record1[1]+record2[1])*50)/games), [record1[0]+record2[2] ,record1[1]+record2[1], record1[2]+record2[0]], games1, games2 


if __name__ == '__main__':
    omodel = tf.keras.models.load_model('saved_models')
    nmodel = tf.keras.models.load_model('models/0')
    score, record, black_games, white_games = eval_model(omodel, nmodel)

    print(score)