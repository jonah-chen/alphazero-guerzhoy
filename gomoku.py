"""Basic Gomoku Program for Project 2. Contains: Game, Basic Computer Player, Basic Scoring Polynomial
Author(s): mak13789, hina
"""

# The tuple (d_y, d_x) represent
# (1,0) direction from left to right (horizontal)
# (0,1) direction from top to bottom (vertical)
# (1,1) direction from upper-left to lower-right 
# #
#   #
#     #
# (1,-1) direction from upper-right to lower left
#     #
#   #
# #


def is_empty(board): 
    '''Return True iff the board is empty
    '''
    for i in board:
        for j in i:
            if ' ' != j:
                return False
    return True
    

def is_bounded(board, y_end, x_end, length, d_y, d_x):
    '''Return 'OPEN' for open sequences, 'SEMIOPEN' for semiopen sequences 
    and 'CLOSED' for closed sequences. Open, Semipoen, and Closed are 
    defined in ESC180H1F
    '''
    # Check (y_end + d_y, x_end + d_x) is empty 
    # or (y_end - length * d_y, x_end - length * d_x) is empty
    y1, x1, y2, x2 = y_end + d_y, x_end + d_x, y_end - length * d_y, x_end - length * d_x
    length = len(board)
    state = 1

    # If one end exceeds the border, no stones can be placed, or if the square is occupied
    # The or short circuits, thus, no index error should be thrown
    if (y1 < 0 or x1 < 0 or y1 >= length or x1 >= length or board[y1][x1] != ' '):
        state -= 1
    if (y2 < 0 or x2 < 0 or y2 >= length or x2 >= length or board[y2][x2] != ' '):
        state -= 1

    # hAhA nO SwitCH StATeMenT iN pYThoN
    if state == 1:
        return 'OPEN'
    if state == 0:
        return 'SEMIOPEN'
    if state == -1:
        return 'CLOSED'

    return 'ERROR!'


def detect_row(board, col, y_start, x_start, length, d_y, d_x):
    '''Return a tuple whose first element is the number of open sequences 
    of color col of length length in the row R and the second element is 
    the number of semiopen sequences of length length in row R
    '''
    open_seq_count, semi_open_seq_count = 0, 0
    
    # starts at (y_start, x_start) and goes in the direciton (d_y, d_x)
    dir = -1
    if (x_start == 0 or y_start == 0):
        dir = 1
    
    board_size = len(board)

    # Create the stuff for the row R and puts it into a list in order. 
    # Cannot use ndarray :( because they are not allowed
    R = []
    x_counter, y_counter = x_start, y_start
    while (x_counter >= 0 and y_counter >= 0 and x_counter < board_size and y_counter < board_size):
        R.append(board[y_counter][x_counter])
        y_counter += dir * d_y
        x_counter += dir * d_x

    # Checks for errors
    if len(R) <= length:
        return (0, 0)

    # Checks for sequences at the edges
    if R[:length] == [col] * length and R[length] == ' ':
        semi_open_seq_count += 1
    if R[-length:] == [col] * length and R[-length - 1] == ' ':
        semi_open_seq_count += 1

    # iterate through R
    for w in range(1, len(R) - length - 1):
        if R[w:w + length] == [col] * length: 
            # Check open sequences
            if R[w - 1] == ' ' and R[w + length] == ' ': 
                open_seq_count += 1
            # Check semi open sequences 
            elif (R[w - 1] == ' ' or R[w + length] == ' ') and R[w + length] != col and R[w - 1] != col:
                semi_open_seq_count += 1

    return open_seq_count, semi_open_seq_count


def detect_rows(board, col, length):
    '''Return a tuple whose first element is the number of open sequences 
    of color col and length length on the entire board, and whose second 
    element is the number of semi-open sequences of color col and length 
    length on the entire board.
    '''
    open_seq_count, semi_open_seq_count = 0, 0
    board_length = len(board)

    for w in range(board_length):
        # Checks for up to down sequences
        x1, x2 = detect_row(board, col, 0, w, length, 1, 0)
        open_seq_count += x1
        semi_open_seq_count += x2
        
        # Checks for left to right sequences
        x1, x2 = detect_row(board, col, w, 0, length, 0, 1)
        open_seq_count += x1
        semi_open_seq_count += x2

        # Checks for diagonal sequences like
        # X
        #  X
        #   X
        x1, x2 = detect_row(board, col, 0, w, length, 1, 1)
        open_seq_count += x1
        semi_open_seq_count += x2
        if w != 0:
            x1, x2 = detect_row(board, col, w, 0, length, 1, 1)
            open_seq_count += x1
            semi_open_seq_count += x2

        # Checks for diagonal sequences like
        #   X
        #  X
        # X
        x1, x2 = detect_row(board, col, w, 0, length, -1, 1)
        open_seq_count += x1
        semi_open_seq_count += x2
        if w != 0:
            x1, x2 = detect_row(board, col, board_length - 1, w, length, 1, -1)
            open_seq_count += x1
            semi_open_seq_count += x2

    return open_seq_count, semi_open_seq_count


def search_max(board):
    move_y, move_x = -1, -1
    max_score = -11111111111111111111111111111111111111111111111
    for i in range(8):
        for j in range(8):
            if (board[i][j] == ' '):
                board[i][j] = 'b'
                if iswin(board) == 1:
                    return i, j
                s = score(board)
                if (s > max_score):
                    move_y, move_x = i, j
                    max_score = s
                board[i][j] = ' '
    return move_y, move_x


def score(board): # return int
    '''Basic scoring polynomial returns int score. Higher score is better for black.'''
    MAX_SCORE = 100000
    
    open_b = {}
    semi_open_b = {}
    open_w = {}
    semi_open_w = {}
    
    for i in range(2, 6):
        open_b[i], semi_open_b[i] = detect_rows(board, "b", i)
        open_w[i], semi_open_w[i] = detect_rows(board, "w", i)
        
    
    if open_b[5] >= 1 or semi_open_b[5] >= 1:
        return MAX_SCORE
    
    elif open_w[5] >= 1 or semi_open_w[5] >= 1:
        return -MAX_SCORE
        
    return (-10000 * (open_w[4] + semi_open_w[4])+ 
            500  * open_b[4]                     + 
            50   * semi_open_b[4]                + 
            -100  * open_w[3]                    + 
            -30   * semi_open_w[3]               + 
            50   * open_b[3]                     + 
            10   * semi_open_b[3]                +  
            open_b[2] + semi_open_b[2] - open_w[2] - semi_open_w[2])


def iswin(board):
    '''Return the index corresponding to the game state in this array
    ["White won", "Black won", "Draw", "Continue Playing"]'''
    draw = True
    for i in range(len(board)):
        for j in range(len(board)):
            if draw and board[i][j] == ' ':
                draw = False
            # horizontal case:
            if j + 4 < len(board):
                temp_1 = []
                for b in range(5):
                    temp_1.append(board[i][j+b])
                if (temp_1 == ["b"] * 5):
                    return 1
                if (temp_1 == ["w"] * 5):
                    return 0
            

            # vertical case:
            if i + 4 < len(board):
                temp_2 = []
                for c in range(5):
                    temp_2.append(board[i+c][j])
                if (temp_2 == ["b"] * 5):
                    return 1
                if (temp_2 == ["w"] * 5):
                    return 0

            # diagonal cases:
            # first case: increasing the row number and the column number:
            if i + 4 < len(board) and j + 4 < len(board):
                temp = []
                for a in range(5):
                    temp.append(board[i+a][j+a])
                if (temp == ["b"] * 5):
                    return 1
                if (temp == ["w"] * 5):
                    return 0
            # second case: increasing the row number but decreasing the column number:
            if i + 4 < len(board) and j - 4 < len(board):
                temp_3 = []
                for d in range(5):
                    temp_3.append(board[i+d][j-d])
                if (temp_3 == ["b"] * 5):
                    return 1
                if (temp_3 == ["w"] * 5):
                    return 0
    
    if draw:
        return 2
    return 3


def is_win(board):
    states = ["White won", "Black won", "Draw", "Continue Playing"]
    return states[iswin(board)]


def print_board(board): # return void
    
    s = "*"
    for i in range(len(board[0])-1):
        s += str(i%10) + "|"
    s += str((len(board[0])-1)%10)
    s += "*\n"
    
    for i in range(len(board)):
        s += str(i%10)
        for j in range(len(board[0])-1):
            s += str(board[i][j]) + "|"
        s += str(board[i][len(board[0])-1]) 
    
        s += "*\n"
    s += (len(board[0])*2 + 1)*"*"
    
    print(s)


def make_empty_board(sz):
    board = []
    for _ in range(sz):
        board.append([" "]*sz)
    return board


def analysis(board):
    # Score
    ## REMOVE THIS OTHERWISE WE WILL GET ZERO!!!!
    print(f'Score: {score(board)}')
    for c, full_name in [["b", "Black"], ["w", "White"]]:
        print("%s stones" % (full_name))
        for i in range(2, 6):
            open, semi_open = detect_rows(board, c, i)
            print("Open rows of length %d: %d" % (i, open))
            print("Semi-open rows of length %d: %d" % (i, semi_open))


def play_gomoku(board_size):
    board = make_empty_board(board_size)
    board_height = len(board)
    board_width = len(board[0])
    
    while True:
        print_board(board)
        if is_empty(board):
            move_y = board_height // 2
            move_x = board_width // 2
        else:
            move_y, move_x = search_max(board)
            
        print("Computer move: (%d, %d)" % (move_y, move_x))
        board[move_y][move_x] = "b"
        print_board(board)
        analysis(board)
        
        game_res = is_win(board)
        if game_res in ["White won", "Black won", "Draw"]:
            return game_res
            
   
        print("Your move:")
        move_y = int(input("y coord: "))
        move_x = int(input("x coord: "))
        board[move_y][move_x] = "w"
        print_board(board)
        analysis(board)
        
        game_res = is_win(board)
        if game_res in ["White won", "Black won", "Draw"]:
            return game_res


def put_seq_on_board(board, y, x, d_y, d_x, length, col):
    for _ in range(length):
        board[y][x] = col        
        y += d_y
        x += d_x


def test_is_empty():
    board  = make_empty_board(8)
    if is_empty(board):
        print("TEST CASE for is_empty PASSED")
    else:
        print("TEST CASE for is_empty FAILED")


def test_is_bounded():
    board = make_empty_board(8)
    x = 5; y = 1; d_x = 0; d_y = 1; length = 3
    put_seq_on_board(board, y, x, d_y, d_x, length, "w")
    print_board(board)
    
    y_end = 3
    x_end = 5

    if is_bounded(board, y_end, x_end, length, d_y, d_x) == 'OPEN':
        print("TEST CASE for is_bounded PASSED")
    else:
        print("TEST CASE for is_bounded FAILED")


def test_detect_row():
    board = make_empty_board(8)
    x = 5; y = 1; d_x = 0; d_y = 1; length = 3
    put_seq_on_board(board, y, x, d_y, d_x, length, "w")
    print_board(board)
    if detect_row(board, "w", 0,x,length,d_y,d_x) == (1,0):
        print("TEST CASE for detect_row PASSED")
    else:
        print("TEST CASE for detect_row FAILED")


def test_detect_rows():
    board = make_empty_board(8)
    x = 5; y = 1; d_x = 0; d_y = 1; length = 3; col = 'w'
    put_seq_on_board(board, y, x, d_y, d_x, length, "w")
    print_board(board)
    if detect_rows(board, col,length) == (1,0):
        print("TEST CASE for detect_rows PASSED")
    else:
        print("TEST CASE for detect_rows FAILED")


def test_search_max():
    board = make_empty_board(8)
    x = 5; y = 0; d_x = 0; d_y = 1; length = 4; col = 'w'
    put_seq_on_board(board, y, x, d_y, d_x, length, col)
    x = 6; y = 0; d_x = 0; d_y = 1; length = 4; col = 'b'
    put_seq_on_board(board, y, x, d_y, d_x, length, col)
    print_board(board)
    if search_max(board) == (4,6):
        print("TEST CASE for search_max PASSED")
    else:
        print("TEST CASE for search_max FAILED")


def easy_testset_for_main_functions():
    test_is_empty()
    test_is_bounded()
    test_detect_row()
    test_detect_rows()
    test_search_max()


def some_tests():
    board = make_empty_board(8)

    board[0][5] = "w"
    board[0][6] = "b"
    y = 5; x = 2; d_x = 0; d_y = 1; length = 3
    put_seq_on_board(board, y, x, d_y, d_x, length, "w")
    print_board(board)
    analysis(board)
    
    # Expected output:
    #       *0|1|2|3|4|5|6|7*
    #       0 | | | | |w|b| *
    #       1 | | | | | | | *
    #       2 | | | | | | | *
    #       3 | | | | | | | *
    #       4 | | | | | | | *
    #       5 | |w| | | | | *
    #       6 | |w| | | | | *
    #       7 | |w| | | | | *
    #       *****************
    #       Black stones:
    #       Open rows of length 2: 0
    #       Semi-open rows of length 2: 0
    #       Open rows of length 3: 0
    #       Semi-open rows of length 3: 0
    #       Open rows of length 4: 0
    #       Semi-open rows of length 4: 0
    #       Open rows of length 5: 0
    #       Semi-open rows of length 5: 0
    #       White stones:
    #       Open rows of length 2: 0
    #       Semi-open rows of length 2: 0
    #       Open rows of length 3: 0
    #       Semi-open rows of length 3: 1
    #       Open rows of length 4: 0
    #       Semi-open rows of length 4: 0
    #       Open rows of length 5: 0
    #       Semi-open rows of length 5: 0
    
    y = 3; x = 5; d_x = -1; d_y = 1; length = 2
    
    put_seq_on_board(board, y, x, d_y, d_x, length, "b")
    print_board(board)
    analysis(board)
    
    # Expected output:
    #        *0|1|2|3|4|5|6|7*
    #        0 | | | | |w|b| *
    #        1 | | | | | | | *
    #        2 | | | | | | | *
    #        3 | | | | |b| | *
    #        4 | | | |b| | | *
    #        5 | |w| | | | | *
    #        6 | |w| | | | | *
    #        7 | |w| | | | | *
    #        *****************
    #
    #         Black stones:
    #         Open rows of length 2: 1
    #         Semi-open rows of length 2: 0
    #         Open rows of length 3: 0
    #         Semi-open rows of length 3: 0
    #         Open rows of length 4: 0
    #         Semi-open rows of length 4: 0
    #         Open rows of length 5: 0
    #         Semi-open rows of length 5: 0
    #         White stones:
    #         Open rows of length 2: 0
    #         Semi-open rows of length 2: 0
    #         Open rows of length 3: 0
    #         Semi-open rows of length 3: 1
    #         Open rows of length 4: 0
    #         Semi-open rows of length 4: 0
    #         Open rows of length 5: 0
    #         Semi-open rows of length 5: 0
    #     
    
    y = 5; x = 3; d_x = -1; d_y = 1; length = 1
    put_seq_on_board(board, y, x, d_y, d_x, length, "b")
    print_board(board)
    analysis(board)   # WHY ARE THERE SEMISCOLONS!!!!!!!!!!!!!!!!!!!!!!!!
    
    #        Expected output:
    #           *0|1|2|3|4|5|6|7*
    #           0 | | | | |w|b| *
    #           1 | | | | | | | *
    #           2 | | | | | | | *
    #           3 | | | | |b| | *
    #           4 | | | |b| | | *
    #           5 | |w|b| | | | *
    #           6 | |w| | | | | *
    #           7 | |w| | | | | *
    #           *****************
    #        
    #        
    #        Black stones:
    #        Open rows of length 2: 0
    #        Semi-open rows of length 2: 0
    #        Open rows of length 3: 0
    #        Semi-open rows of length 3: 1
    #        Open rows of length 4: 0
    #        Semi-open rows of length 4: 0
    #        Open rows of length 5: 0
    #        Semi-open rows of length 5: 0
    #        White stones:
    #        Open rows of length 2: 0
    #        Semi-open rows of length 2: 0
    #        Open rows of length 3: 0
    #        Semi-open rows of length 3: 1
    #        Open rows of length 4: 0
    #        Semi-open rows of length 4: 0
    #        Open rows of length 5: 0
    #        Semi-open rows of length 5: 0


if __name__ == '__main__':
    test_is_bounded()
