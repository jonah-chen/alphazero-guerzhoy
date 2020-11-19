from concurrent.futures import ProcessPoolExecutor

from nptrain import is_win
from ai import Game
import numpy as np
import time
import random

def generate():
    final_boards = []
    states = []
    for _ in range(100000):
        G = Game()
        i = 0
        while G.is_win() == 0:
            while(G.move(int(random.random()*8), int(random.random()*8), i%2+1) == 1):0
            i += 1
        final_boards.append(str(G))
        states.append(G.is_win())
    return final_boards, states
        


start = time.perf_counter()

with ProcessPoolExecutor() as executor:
    results = [executor.submit(generate) for _ in range(12)]

end = time.perf_counter()
print(end-start)

results_list = ["Continue Playing", "Black won", "White won", "Draw"]
while True:
    i = int(input("Enter the game you want to see(0,99999)"))
    t = int(input("Enter the thread (0-11)"))
    boards, states = results[t].result()
    print(results_list[states[i]] + '\n')
    print(boards[t])
        