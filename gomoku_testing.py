from gomoku import print_board, is_empty, detect_rows, search_max, is_win
import numpy as np
import random
def gen_random_board():
    temp = [[' '] * 8 for c in range(8)]
    for a in range(len(temp)):
        for b in range(len(temp[a])):
            temp_1 = [' ', 'b', 'w']
            temp[a][b] = random.choice(temp_1)
    return temp


def test_gomoku(n):
    for d in range(n):
        new_board = gen_random_board()
        print_board(new_board)
        print ("is_empty test: " + str(is_empty(new_board)))
        for color in ['w', 'b']:
            for e in range(2, 9):
                temp_5 = detect_rows(new_board, color, e)
                print ("detect_rows test with color " + color + " and length " + str(e) +  ": " + str(temp_5))
        print ("search_max test: " + str(search_max(new_board)))
        print ("is_win test: " + is_win(new_board))
        for f in range(7):
            print(" ")
test_gomoku(20)              