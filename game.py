import numpy as np
from nptrain import is_win


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
        '''Return the game state as the index in the array ["Continue Playing", "Black won", "White won", "Draw"]'''
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
    
    def __str__(self):
        s = "  0"
        for i in range(1,8):
            s += f"|{i}"
        
        for i in range(8):
            s += f"\n{i}|" 
            for j in range(8):
                if self.black[i,j,0]:
                    s += "b "
                elif self.white[i,j,0]:
                    s += "w "
                else:
                    s += "  "
        return s
