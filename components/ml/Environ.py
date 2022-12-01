import pandas as pd
import numpy as np
import chess


class Environ:
    """
        This class manages the chessboard and determines what the state is.
        The class passes the state to the agent. 
        
        after the agent(s) is trained, this class will be used to manage the chessboard.
    """
    def __init__(self, chess_data):
        self.chess_data = chess_data 
        self.board = chess.Board()
        self.turn_list = self.chess_data.columns[5:155].tolist()
        self.turn_index = 0 
        self.legal_moves = self.get_legal_moves()
        
    
    def get_curr_state(self):
        """ the environ class is responsible for determining the state at all times
            all changes to the chessboard and the state are handled by a method of the environ class
            :param none
            :return a dictionary that defines the current state that an agent will act on
        """
        return {'turn_index': self.turn_index, 'turn_list': self.turn_list, 'legal_moves': self.legal_moves}
    
    
    def update_curr_state(self):
        """ current state is the current turn and the legal moves at that turn 
            the state is updated each time a chess_move str is loaded to the chessboard.
        """
        self.turn_index += 1
        self.legal_moves = self.get_legal_moves()
        
        
    def get_curr_turn(self):                        
        """ returns the string of the current turn, 'W2' for example
            which would correspond to index = 2
            :param none
            :return, a string that corresponds to current turn
        """
        return self.turn_list[self.turn_index]
    
    
    def load_chessboard(self, chess_move_str):
        """ method to play move on chessboard. call this method when you want to commit a chess move.
            the agent class chooses the move, but the environ class must load up the chessboard with that move
            :param chess_move as string
            :return bool for success or failure.
        """
        try:
            self.board.push_san(chess_move_str)
            return True
        except ValueError:
            return False
        
    
    def reset_environ(self):
        """ resets environ, call each time after a game ends, and after training period is over.
            :param none
            :return none
        """
        self.board.reset()
        self.turn_index = 0
        self.legal_moves = self.get_legal_moves()
    
    
    def get_legal_moves(self):   
        """ method will return a list of strings that represents the legal moves at that turn
            :param none
            :return list of strings that represents the legal moves at a turn, given the board state
        """
        legal_moves = []
        for i in self.board.legal_moves:
            legal_moves.append(i)

        i = 0
        num_legal_moves = self.board.legal_moves.count()
        legal_moves_str = []
        while num_legal_moves > 0:
            legal_moves_str.append(self.board.san(legal_moves[i]))
            i += 1
            num_legal_moves -= 1
        
        return legal_moves_str