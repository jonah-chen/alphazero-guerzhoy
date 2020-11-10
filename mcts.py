import numpy as np

class Node:
    '''Node of the tree for MCTS'''
    def __init__(self, parent):
        '''initialization function for this class'''
        self.P = 0.0
        self.Q = 0.0
        self.N = 0.0
        self.U = 0.0
        self.V = 0.0