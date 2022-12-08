###

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


    def get_curr_state(self):
        """ the environ class is responsible for determining the state at all times
            all changes to the chessboard and the state are handled by a method of the environ class
            :param none
            :return a dictionary that defines the current state that an agent will act on
        """
        return {'turn_index': self.turn_index, 'curr_turn': self.get_curr_turn(), 'legal_moves': self.get_legal_moves()}
    
    
    def update_curr_state(self):
        """ current state is the current turn and the legal moves at that turn 
            the state is updated each time a chess_move str is loaded to the chessboard.
        """
        self.turn_index += 1

        
    def get_curr_turn(self):                        
        """ returns the string of the current turn, 'W2' for example
            which would correspond to index = 2
            :param none
            :return, a string that corresponds to current turn
        """
        try: 
            curr_turn = self.turn_list[self.turn_index]
            return curr_turn
        except IndexError:
            print(f'list index out of range, turn index is {self.turn_index}')
            return False
    
    
    def load_chessboard(self, chess_move_str):
        """ method to play move on chessboard. call this method when you want to commit a chess move.
            the agent class chooses the move, but the environ class must load up the chessboard with that move
            :param chess_move as string like this, 'Nf3'
            :return bool for success or failure.
        """
        try:
            self.board.push_san(chess_move_str)
            return True
        except ValueError:
            return False
    

    def pop_chessboard(self):
        """ pops the most recent move applied to the chessboard"""
        self.board.pop()


    def load_chessboard_for_Q_est(self, analysis_results):
        """ method should only be called during training. this will load the 
            chessboard using a Move.uci string.
            :pre analysis of proposed board condition has already been done
            :post board is loaded with black's anticipated move
            :param q_est_board_analysis_results, 
            it has this form, [{'mate_score': <some score>, 'centipawn_score': <some score>, 'anticipated_next_move': <move>}]
            :return bool for success or failure
        """
        chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        try:
            self.board.push(chess_move)
            return True
        except ValueError:
            print('failed to add antiipated move')
            return False
    
    def reset_environ(self):
        """ resets environ, call each time after a game ends, and after training period is over.
            :param none
            :return none
        """
        self.board.reset()
        self.turn_index = 0

    
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