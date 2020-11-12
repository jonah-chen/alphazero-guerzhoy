import numpy as np
from nptrain import *
from copy import copy, deepcopy
import tensorflow as tf
from ai import LOC
from game import get_prob, move_on_board
from concurrent.futures import ProcessPoolExecutor

from time import perf_counter
import random
from game import print_board
# AlphaZero Guerzhoy

C_PUCT = 1.0
TAU = 1.0
ALPHA = 0.03
EPSILON = 0.25

class Node:
    '''Node of the tree for MCTS'''
    def __init__(self, p_a, player):
        '''ini  tialization function for this class
        parent: Node
        p_a: Probability of the action
        action: (y,x) signifying a legal move.'''
        self.N = 0
        self.W = 0
        self.P = p_a
        self.player = player
        self.expanded = False
        self.children = {}


    def __str__(self):
        s = f"Q:{self.Q()}, N:{self.N}, P:{self.P}, num_children:{len(self.children)}, expanded:{self.expanded}, player:{self.player}\n"
        for (key, child) in self.children.items():
            s += f"Action:{key}, N={child.N}, Q+U={minusQ_plus_U(C_PUCT, child.W, child.N, self.N, child.P):.4f}, P={child.P:.4f}\n"
        return s

    def Q(self):
        return 0 if self.N == 0 else self.W / self.N

    def select(self):
        """Select the best action and child based on argmax(Q+U)"""
        action, child, val = -1, None, -np.inf
        for a, c in self.children.items():
            qw = minusQ_plus_U(C_PUCT, c.W, c.N, self.N, c.P)
            if qw > val:
                action = a
                child = c
                val = qw
        return action, child


    def play(self, tau=TAU):
        """Select a move to play based on the result of the MCTS and the temperature parameter tau.
        """
        N = np.array([child.N for child in self.children.values()])
        a = [action for action in self.children.keys()]
        
        p = N ** (1 / tau)
        p = p / np.sum(p)
        if tau == 0:
            action = a[np.argmax(N)]
        else:
            action = np.random.choice(a, p=p)
        
        # Calculate the distribution used for training.
        dist = np.zeros((64,))
        for i in range(len(a)):
            dist[a[i]] = p[i]

        return action, dist
        

    def expand(self, p, board):
        """Expand the leaf node by using the policy vector outputted  by the neural network p as an argument. board is only used to determine legal moves."""
        probs = get_prob(board, p + (1 - EPSILON) * EPSILON * np.random.dirichlet(np.ones(64,))) 

        for i in range(64):
            if probs[i] > 0:
                self.children[i] = Node(probs[i], self.player % 2 + 1)
        self.expanded = True

                                                                                                                                                            
def search_instance(model, board, player, it=1024):
    """Takes the board from player 1 perspective and performs an instance of MCTS.
    """
    start = perf_counter() #
    root = Node(0, player)
    # MUST be batched for training
    policy, value = model.predict(np.array([board if player == 1 else np.flip(board, axis=2)])) 
    root.expand(policy[0], board)

    end = perf_counter() #
    print(f'{1000*(end-start):.2f}ms for root') #

    for _ in range(it):
        print(f'iteration{_}') #
        start = perf_counter() #

        search_board = deepcopy(board)
        
        end = perf_counter() #
        print(f'{1000*(end-start):.2f}ms for deepcopy')
        start = perf_counter() #


        node = root
        path = [node] # Preserves the history for the backup step
        p = player
        while node.expanded:
            action, node = node.select()
            path.append(node)
            move_on_board(search_board, action, player=p)
            p = p % 2 + 1
        
        end = perf_counter() #
        print(f'{1000*(end-start):.2f}ms for select')
        start = perf_counter() #


        state = is_win(search_board)
        if state == 1:
            p1, p2 = 1, -1
            print("black win")
        elif state == 2:
            p1, p2 = -1, 1
            print("white win")
        elif state == 3:
            p1, p2 = 0, 0
            print("draw")
        else:
            if p == 2:
                search_board = np.flip(search_board, axis=2)
            policy, value = model.predict(np.array([search_board])) # MUST be batched for training
            node.expand(policy[0], search_board)
        
        end = perf_counter() #
        print(f'{1000*(end-start):.2f}ms for expand and eval')
        start = perf_counter() #

        # Backup here
        for bnode in path[::-1]:
            bnode.N += 1
            if bnode.player == p:
                bnode.W += value[0,0]
            else:
                bnode.W -= value[0,0]
        
        end = perf_counter() #
        print(f'{1000*(end-start):.2f}ms for backup')

    return root

def optimized_search(model, boards, players, it=1024, roots=None):
    """Perform Monte-Carlo Tree Search on a batch of boards with it iterations. 
    Return a list of Node objects, whose children is selected with with a given probability distribution for the next move.
    """
    games = len(boards)

    # Store all the root nodes that need to be evaluated
    if roots is None:
        roots = [Node(0, player) for player in players]
        # Calculate the policies the root nodes
        # Note the values of the root nodes are irrelevent
        policies, _ = model.predict(np.array([boards[i] if players[i] == 1 else np.flip(boards[i], axis=2) for i in range(games)]))

        # Expand all root nodes
        for i in range(games):
            roots[i].expand(policies[i], boards[i])
    
    start = perf_counter()
    # Performs the search
    for _ in range(it):
        if _ % 32 == 31:
            end = perf_counter()
            print(f'iteration:{_+1} of {it} time:{(end-start):.2f}s')
            start = perf_counter()
            
        search_boards = deepcopy(boards)
        nodes = copy(roots)
        search_players = copy(players)
        paths = [[node] for node in nodes] # numpy transpose may or may not be faster

        # For all games, expands the nodes
        for i in range(games):
            while nodes[i].expanded:
                action, nodes[i] = nodes[i].select()
                paths[i].append(nodes[i])
                move_on_board(search_boards[i], action, player=search_players[i])
                search_players[i] = search_players[i] % 2 + 1
    
        # Flips the boards of player 2 so that it can be evaluated by neural network

        for i in range(games):
            if search_players[i] != 1:
                search_boards[i] = np.flip(search_boards[i], axis=2)
        
        # Evaulate: Determine the policies and values as well and the end state of the leaf nodes.
        policies, values = model.predict(search_boards)
        values = values.flatten()
        states = [is_win(search_board) for search_board in search_boards]

        # Correct the evaluation for games with end states. -1.0 for loss, 0.0 for draw, 1.0 for win.
        for i in range(games):
            if states[i] == 1:
                values[i] = 1.0
            elif states[i] == 2:
                values[i] = -1.0
            elif states[i] == 3:
                values[i] == 0.0
            else:
                nodes[i].expand(policies[i], search_boards[i])

        for i in range(games):
            for bnode in paths[i][::-1]:
                bnode.N += 1
                if bnode.player == search_players[i]:
                    bnode.W += values[i]
                else:
                    bnode.W -= values[i]
    return roots


def self_play(model, games=64, game_iter=64, search_iter=1024):
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




if __name__ == '__main__':
    true_start = perf_counter()
    model = tf.keras.models.load_model(LOC)

    s, pie, z = self_play(model, games=64)
    
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
    
    np.save('selfplay_data/0000/pie', pie)
    np.save('selfplay_data/0000/z', z)
    np.save('selfplay_data/0000/s', s)

    # boards = np.zeros((128,8,8,2,), dtype="float32")
    # players = [1]*128

    # results = None

    # while 1:
    #     results = optimized_search(model, boards, players, roots=results, it=5)
    #     for i in range(128):
    #         if not is_win(boards[i]):
    #             act = results[i].play()
    #             results[i] = results[i].children[act]
    #             move_on_board(boards[i], act, player=players[i])
    #         print(f"Player{i} selected action {act}. The result is {is_win(boards[i])}\n")
    #         print_board(boards[i])
    #         print("\n")
    #         players[i] = players[i] % 2 + 1
    