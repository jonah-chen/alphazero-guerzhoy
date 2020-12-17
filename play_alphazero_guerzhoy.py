from tensorflow.keras.models import load_model
import numpy as np
from game import move_on_board, print_board
from mcts import optimized_search
from tkinter import Tk, Label, Button, DISABLED
from PIL import ImageTk, Image
from nptrain import is_win

root = Tk()
root.title("Play AlphaZero Guerzhoy")

GOOD_MODELS = [0,2,8,9,13,19,24,35,39,58,60,67]
black_stone = ImageTk.PhotoImage(Image.open("images/blackstone.png").resize((100,100)))
white_stone = ImageTk.PhotoImage(Image.open("images/whitestone.png").resize((100,100)))

score = [0, 0] # Your score, Computer Score

def computer_move(model, board):
    """Makes a move by using the A.I. on a board board

    Args:
        model (tf.keras.models.Model): The trained A.I. Model
        board (np.array (shape=(1,8,8,2,))): The board

    Returns:
        void
    """
    move = optimized_search(model, board, [1 if player else 2], it=diff)[0].play(0)[0]
    move_on_board(board[0], move, 1 if player else 2)
    return move//8, move%8



def load(i):
    global model
    model = load_model(f"models/{GOOD_MODELS[i]}.h5")

def play_as_white():
    global player
    player = 0
    try:
        x, y = computer_move(model, board)
    except NameError: 
        Label(root, text="Please choose an A.I. to play against.").grid(row=4, column=9, columnspan=3)
        return
        
    board_buttons[x][y].grid_forget()
    board_buttons[x][y] = Button(root, image=white_stone if player else black_stone, state=DISABLED, padx=50, pady=50)
    board_buttons[x][y].grid(row=x, column=y)

def play_as_black():
    global player
    player = 1

def move(x, y):
    global board_buttons

    try: 
        move_on_board(board[0], x*8+y, 2 if player else 1)
    except NameError: 
        Label(root, text="Please choose to be either white or black.").grid(row=4, column=9, columnspan=3)
        return

    board_buttons[x][y].grid_forget()
    board_buttons[x][y] = Button(root, image=black_stone if player else white_stone, state=DISABLED, padx=50, pady=50)
    board_buttons[x][y].grid(row=x, column=y)
    result = is_win(board[0]) # Check if someone wins
    if result == 2:
        if player:
            score[0] += 1
            Label(root, text=f"Player wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        else:
            score[1] += 1
            Label(root, text=f"Computer wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        return
    if result == 1:
        if player:
            score[1] += 1
            Label(root, text=f"Computer wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        else:
            score[0] += 1
            Label(root, text=f"Player wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        return
    if result == 3:
        score[0] += 0.5
        score[1] += 0.5
        Label(root, text=f"Draw! Reset to play again.").grid(row=4, column=9, columnspan=3)
        return
    
    # The computer makes a move in response.
    try:
        x, y = computer_move(model, board)
    except NameError: 
        Label(root, text="Please choose an A.I. to play against.").grid(row=4, column=9, columnspan=3)
        return
    
    board_buttons[x][y].grid_forget()
    board_buttons[x][y] = Button(root, image=white_stone if player else black_stone, state=DISABLED, padx=50, pady=50)
    board_buttons[x][y].grid(row=x, column=y)

    result = is_win(board[0]) # Check if someone wins
    if result == 2:
        if player:
            score[0] += 1
            Label(root, text=f"Player wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        else:
            score[1] += 1
            Label(root, text=f"Computer wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        return
    if result == 1:
        if player:
            score[1] += 1
            Label(root, text=f"Computer wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        else:
            score[0] += 1
            Label(root, text=f"Player wins! Reset to play again.").grid(row=4, column=9, columnspan=3)
        return
    if result == 3:
        score[0] += 0.5
        score[1] += 0.5
        Label(root, text=f"Draw! Reset to play again.").grid(row=4, column=9, columnspan=3)
        return
    
    
def select_diff(difficulty):
    global diff
    diff = difficulty

def resign():
    global score
    score[1] += 1
    init_game()
    


def init_game():
    global board_buttons, board, diff, player

    try:
        del player
    except NameError:
        pass
    diff = 512

    board_buttons = [[Button(root, command=lambda: move(0, 0), padx=50, pady=50), Button(root, command=lambda: move(0, 1), padx=50, pady=50), Button(root, command=lambda: move(0, 2), padx=50, pady=50), Button(root, command=lambda: move(0, 3), padx=50, pady=50), Button(root, command=lambda: move(0, 4), padx=50, pady=50), Button(root, command=lambda: move(0, 5), padx=50, pady=50), Button(root, command=lambda: move(0, 6), padx=50, pady=50), Button(root, command=lambda: move(0, 7), padx=50, pady=50)], [Button(root, command=lambda: move(1, 0), padx=50, pady=50), Button(root, command=lambda: move(1, 1), padx=50, pady=50), Button(root, command=lambda: move(1, 2), padx=50, pady=50), Button(root, command=lambda: move(1, 3), padx=50, pady=50), Button(root, command=lambda: move(1, 4), padx=50, pady=50), Button(root, command=lambda: move(1, 5), padx=50, pady=50), Button(root, command=lambda: move(1, 6), padx=50, pady=50), Button(root, command=lambda: move(1, 7), padx=50, pady=50)], [Button(root, command=lambda: move(2, 0), padx=50, pady=50), Button(root, command=lambda: move(2, 1), padx=50, pady=50), Button(root, command=lambda: move(2, 2), padx=50, pady=50), Button(root, command=lambda: move(2, 3), padx=50, pady=50), Button(root, command=lambda: move(2, 4), padx=50, pady=50), Button(root, command=lambda: move(2, 5), padx=50, pady=50), Button(root, command=lambda: move(2, 6), padx=50, pady=50), Button(root, command=lambda: move(2, 7), padx=50, pady=50)], [Button(root, command=lambda: move(3, 0), padx=50, pady=50), Button(root, command=lambda: move(3, 1), padx=50, pady=50), Button(root, command=lambda: move(3, 2), padx=50, pady=50), Button(root, command=lambda: move(3, 3), padx=50, pady=50), Button(root, command=lambda: move(3, 4), padx=50, pady=50), Button(root, command=lambda: move(3, 5), padx=50, pady=50), Button(root, command=lambda: move(3, 6), padx=50, pady=50), Button(root, command=lambda: move(3, 7), padx=50, pady=50)], [Button(root, command=lambda: move(4, 0), padx=50, pady=50), Button(root, command=lambda: move(4, 1), padx=50, pady=50), Button(root, command=lambda: move(4, 2), padx=50, pady=50), Button(root, command=lambda: move(4, 3), padx=50, pady=50), Button(root, command=lambda: move(4, 4), padx=50, pady=50), Button(root, command=lambda: move(4, 5), padx=50, pady=50), Button(root, command=lambda: move(4, 6), padx=50, pady=50), Button(root, command=lambda: move(4, 7), padx=50, pady=50)], [Button(root, command=lambda: move(5, 0), padx=50, pady=50), Button(root, command=lambda: move(5, 1), padx=50, pady=50), Button(root, command=lambda: move(5, 2), padx=50, pady=50), Button(root, command=lambda: move(5, 3), padx=50, pady=50), Button(root, command=lambda: move(5, 4), padx=50, pady=50), Button(root, command=lambda: move(5, 5), padx=50, pady=50), Button(root, command=lambda: move(5, 6), padx=50, pady=50), Button(root, command=lambda: move(5, 7), padx=50, pady=50)], [Button(root, command=lambda: move(6, 0), padx=50, pady=50), Button(root, command=lambda: move(6, 1), padx=50, pady=50), Button(root, command=lambda: move(6, 2), padx=50, pady=50), Button(root, command=lambda: move(6, 3), padx=50, pady=50), Button(root, command=lambda: move(6, 4), padx=50, pady=50), Button(root, command=lambda: move(6, 5), padx=50, pady=50), Button(root, command=lambda: move(6, 6), padx=50, pady=50), Button(root, command=lambda: move(6, 7), padx=50, pady=50)], [Button(root, command=lambda: move(7, 0), padx=50, pady=50), Button(root, command=lambda: move(7, 1), padx=50, pady=50), Button(root, command=lambda: move(7, 2), padx=50, pady=50), Button(root, command=lambda: move(7, 3), padx=50, pady=50), Button(root, command=lambda: move(7, 4), padx=50, pady=50), Button(root, command=lambda: move(7, 5), padx=50, pady=50), Button(root, command=lambda: move(7, 6), padx=50, pady=50), Button(root, command=lambda: move(7, 7), padx=50, pady=50)]]

    for i in range(8):
        for j in range(8):
            board_buttons[i][j].grid(row=i, column=j)

    Button(root, text="RESET", command=init_game).grid(row=8, column=0)
    Button(root, text="Resign", command=resign).grid(row=8, column=1)
    Button(root, text="Black", command=play_as_black).grid(row=8, column=3)
    Button(root, text="White", command=play_as_white).grid(row=8, column=4)

    model_select = [Button(root, text=f"{i}", command=lambda: load(i)) for i in range(len(GOOD_MODELS)-1)]
    model_select.append(Button(root, text="BETA", command=lambda: load(-1)))

    for i in range(len(GOOD_MODELS)):
        model_select[i].grid(row=9, column=i)

    Button(root, text="Easy", command=lambda: select_diff(128)).grid(row=8, column=6)
    Button(root, text="Default", command=lambda: select_diff(512)).grid(row=8, column=7)
    Button(root, text="Hard", command=lambda: select_diff(2048)).grid(row=8, column=8)

    Label(root, text=f"Your score: {score[0]}").grid(row=0, column=9, columnspan=3)
    Label(root, text=f"Computer score: {score[1]}").grid(row=1, column=9, columnspan=3)

    board = np.zeros((1, 8, 8, 2,), dtype="float32")

if __name__ == "__main__":
    init_game()
    root.mainloop()
