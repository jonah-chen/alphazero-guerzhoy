import numpy as np

from game import move_on_board
from debug import convert_good_to_bad_board
from gomoku import score, make_empty_board

from tkinter import Tk, Button, Label, Entry, DISABLED
import tensorflow as tf

from PIL import Image, ImageTk



class Analysis:
    def __init__(self, guerzhoy_analysis=True, model_analysis=True):
        global white_stone, black_stone, blank
        self.root = Tk()
        
        
        black_stone = ImageTk.PhotoImage(Image.open("images/blackstone.png").resize((130,130)))
        white_stone = ImageTk.PhotoImage(Image.open("images/whitestone.png").resize((130,130)))
        blank = ImageTk.PhotoImage(Image.new('RGB', (100, 100), color="white"))
        
        # Create the board to analyze with score
        if guerzhoy_analysis:
            self.analysis_board = make_empty_board(8)
            Label(self.root, text="Guerzhoy").grid(row=9, column=5)
            self.evaluation = Label(self.root, text="N")
            self.evaluation.grid(row=10, column=5)
        
        if model_analysis:
            self.analysis_board_2 = np.zeros((8,8,2,), dtype=np.float32)
            self.model_policy = [[Label(self.root, text="N", image=blank) for __ in range(8)] for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    self.model_policy[i][j].grid(row=i, column=j+10, padx=15, pady=15)
        
        # Creates the GUI for the board
        self.board = [[Label(self.root, image=blank) for __ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                self.board[i][j].grid(row=i, column=j, padx=15, pady=15)

        
        # Create game selection between black player and white player.
        self.black_player = Entry(self.root)
        self.white_player = Entry(self.root)
        
        Label(self.root, text="Black").grid(row=9, column=0)
        Label(self.root, text="White").grid(row=9, column=1)
        Button(self.root, text="Load Games", command=self.load_games).grid(row=9, column=2)
        self.black_player.grid(row=10, column=0)
        self.white_player.grid(row=10, column=1)

        # Create the button to incrament the move.
        Button(self.root, text="MOVE!", command=self.next_turn).grid(row=10, column=3)
        Button(self.root, text="End State", command=self.end_game).grid(row=9, column=3)

        # Create reset button.
        Button(self.root, text="RESET", command=self.reset_boards).grid(row=11, column=3)

        # Create model selection for analysis
        Label(self.root, text="Analysis No.").grid(row=9, column=10)
        Button(self.root, text="Load Model", command=self.load_model).grid(row=10, column=11)
        self.model_no = Entry(self.root) # Entry for the model number.
        self.model_no.grid(row=10, column=10)
        self.analysis_model = None

        Label(self.root, text="Model eval.").grid(row=9, column=14)
        
        self.model_eval = Label(self.root, text="N")
        self.model_eval.grid(row=10, column=14)


        # Create the games. The games must be selected using the button "Load Games"
        self.games = None
        self.game_no = 0
        self.turn_no = 0
        Button(self.root, text="<", command=self.prev_game).grid(row=9, column=6)
        Button(self.root, text=">", command=self.next_game).grid(row=9, column=7)


    def reset_boards(self):
        """Resets all boards.
        """

        # Resets the turn number
        self.turn_no = 0

        # Reset the show board
        for i in range(8):
            for j in range(8):
                self.board[i][j].grid_forget()
        self.board = [[Label(self.root, image=blank) for __ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                self.board[i][j].grid(row=i, column=j, padx=15, pady=15)

        # Reset Guerzhoy Analysis board

        try:
            self.analysis_board = make_empty_board(8)
            self.evaluation.grid_forget()
            self.evaluation = Label(self.root, text="N")
            self.evaluation.grid(row=10, column=5)
        except NameError:
            pass

        # Reset model analysis board
        try:
            self.analysis_board_2 = np.zeros((8,8,2,), dtype=np.float32)
            for i in range(8):
                for j in range(8):
                    self.model_policy[i][j].grid_forget()
            self.model_policy = [[Label(self.root, image=blank) for __ in range(8)] for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    self.model_policy[i][j].grid(row=i, column=j+10, padx=15, pady=15)
        except NameError:
            pass
            
        



    def play(self, move, color):
        """Plays a move for analysis

        Args:
            move (int): integer from 0 to 63 representing a legal move
            color (string): 'w' or 'b' for white and black player
        """
        i, j = move // 8, move % 8 # not good practice but I'm lazy

        # Plays the move on the "showboard"
        self.board[i][j].grid_forget()
        self.board[i][j] = Label(self.root, image=white_stone if color=="white" else black_stone)
        self.board[i][j].grid(row=i, column=j)
        
        # Plays the move on analysis board
        try:
            self.analysis_board[i][j] = "w" if color=="white" else "b"

            self.evaluation.grid_forget()
            self.evaluation = Label(self.root, text=f"{score(self.analysis_board)}")
            self.evaluation.grid(row=10, column=5)
            
        except NameError:
            pass
        
        # Play the move on model analysis board
        try:
            move_on_board(self.analysis_board_2, move, 2 if color=="white" else 1)
            if color=="white":
                policy, value = self.analysis_model(np.flip(self.analysis_board_2, axis=2).reshape(1,8,8,2,), training=False)
                self.model_eval.grid_forget()
                self.model_eval = Label(self.root, text=f"{int(1000*value[0])}m")
                self.model_eval.grid(row=10, column=14)
            else:
                policy, value = self.analysis_model(np.reshape(self.analysis_board_2,(1,8,8,2,)), training=False)
                self.model_eval.grid_forget()
                self.model_eval = Label(self.root, text=f"{int(-1000*value[0])}m")
                self.model_eval.grid(row=10, column=14)
            policy = policy.numpy()[0]
            
            for i in range(8):
                for j in range(8):
                    self.model_policy[i][j].grid_forget()
                    self.model_policy[i][j] = Label(self.root, text=f"{100*policy[i*8+j]:.2f}%", bg=(("#0" if int(256*policy[i*8+j]) < 16 else "#") + hex(int(256*policy[i*8+j]))[2:] + "0000"), fg="white", font="Times 28 bold")
                    self.model_policy[i][j].grid(row=i, column=j+10, padx=15, pady=15)
        except NameError:
            print("Model for analysis is not loaded.")
        except ValueError:
            print("Illegal move has been made.")
        

    def load_games(self):
        try:
            self.games = np.load(f"games/{self.black_player.get()}v{self.white_player.get()}.npy", allow_pickle=True)
        except FileNotFoundError:
            print("Cannot find games.")


    def load_model(self):
        try:
            self.analysis_model = tf.keras.models.load_model(f"models/{self.model_no.get()}.h5")
        except FileNotFoundError:
            print("The model selected is invalid.")

    
    def prev_game(self):
        if self.games is None:
            print("No games are loaded.")
            return
        if self.game_no == 0:
            print("Already on the first game.")
            return
        self.game_no -= 1
        self.reset_boards()
        self.next_turn()
        
    def next_game(self):
        if self.games is None:
            print("No games are loaded.")
            return
        self.reset_boards()
        self.game_no += 1
        try:
            self.next_turn()
        except IndexError:
            self.game_no -= 1
            print("Already the last game.")
        
            
    
    def next_turn(self):
        if self.games is None:
            print("No games are loaded.")
            return False
        if self.turn_no >= len(self.games[self.game_no]):
            print("Game has ended.")
            return False

        # Fetches the move
        move = self.games[self.game_no][self.turn_no]
        color = "black" if self.turn_no % 2 == 0 else "white"
        self.play(move, color)

        self.turn_no += 1
        return True

    def end_game(self):
        while self.next_turn():
            pass


        


if __name__ == '__main__':
    a = Analysis()
    a.root.mainloop()

