import numpy as np
from nptrain import is_win

# Depricate this for the following representation
#
# Board represented with numpy array shape (8,8,2,)
# np.flip(board, axis=2) will flip the perspective of the board for the other player
# 
# The symmetries of the board are:
# board
# np.rot90(board, 1, (0,1,))
# np.rot90(board, 2, (0,1,)) 
# np.rot90(board, 3, (0,1,))
#
# The four above action followed by np.flip(board, axis=0)


def check_legal_actions(board):
    """Return the legal actions like that returned by the policy head in shape (64,)
    """
    return board[:,:,0]==board[:,:,1].reshape(64,)

def get_prob(board, policy_val):
    """Return the probabilities of selecting the children of a node given the policy 
    and removing illegal moves"""
    x = (board[:,:,0] == board[:,:,1]).reshape(64,) * policy_val
    return x / np.sum(x)

def check_legal_moves(board):
    """Return the legal moves on the board.
    """
    return board[:,:,0]==board[:,:,1]

def move_on_board(board, move, player=1, takeback=0):
    """Make a move for player player, or for yourself if no player argument is given.
     Dangerous function may cause illegal board states"""
    board[move // 8, move % 8, player - 1] = 0.0 if takeback else 1.0 

def print_board(board):
    s = "  0"
    for i in range(1,8):
        s += f"|{i}"
    
    for i in range(8):
        s += f"\n{i}|" 
        for j in range(8):
            if board[i,j,0]:
                s += u"\u03b4"
            elif board[i,j,1]:
                s += u"\u03b5"
            else:
                s += " "
            if j != 7:
                s += "|"
    print(s)


class Game:
    def __init__(self, black=np.zeros((8,8,2,), dtype=np.float32), white=np.zeros((8,8,2,), dtype=np.float32)):
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
        if (player != 1 and player != 2) or self.black[y, x, 0] or self.black[y, x, 1]:
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
                    s += "b"
                elif self.white[i,j,0]:
                    s += "w"
                else:
                    s += " "
                if j != 7:
                    s += "|"
        return s

if __name__ == "__main__":
    board = np.zeros((8,8,2,))
    move_on_board(board, 3, player=1)
    move_on_board(board, 11, player=2)
    print_board(board)
