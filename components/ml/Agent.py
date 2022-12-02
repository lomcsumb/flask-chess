###
# import Settings
from components.ml.Settings import Settings
import pandas as pd
import numpy as np
import random
import math 
from functools import wraps
import time
import re
# from helper_methods import *
from components.ml.helper_methods import *
import copy
import random

class Agent:
    """ The agent class is responsible for deciding what chess move to play 
        based on the current state. The state is passed the the agent by the environ class.
        The agent class does not make any changes to the chessboard.
    """
    def __init__(self, color, chess_data, rand_game, is_opp_agent = False):        
        self.color = color
        self.chess_data = chess_data
        self.settings = Settings()
        self.is_trained = False 
        self.curr_game = rand_game 
        self.is_opp_agent = is_opp_agent
        self.Q_table = self.init_Q_table(self.chess_data)

    # @timeit
    def choose_next_action(self, environ_state):
        """ this method finds the very next chess move in the curr game sequence, this
            method is ONLY used during training 
        """
        next_turn_index = environ_state['turn_index'] + 1
        turn_list = environ_state['turn_list']
        next_turn = turn_list[next_turn_index]
        chess_move_str = self.chess_data.at[self.curr_game, next_turn]
        chess_move_val = self.chess_data.at[self.curr_game, next_turn + '-Numeric']
        return {'chess_move_str': chess_move_str, 'chess_move_val': chess_move_val}

    
    def choose_action_move_pts_calc(self, curr_game, environ_state):
        """ method is only for finding the move value given by stockfish analysis
        """
        self.turn_index = environ_state['turn_index']
        self.turn_list = environ_state['turn_list']
        self.curr_turn = self.turn_list[self.turn_index]
        self.curr_game = curr_game

        chess_move_str = self.chess_data.at[self.curr_game, self.curr_turn]
        return {'chess_move_str': chess_move_str}


    # @timeit
    def choose_action(self, environ_state):
        """ method does two things. First, this method helps to train the agent. Once the agent is trained, 
            this method helps to pick the appropriate move based on highest val in the Q table for a given turn
            :param environ_state is a dictionary 
            :return a dict containing chess move
        """
        self.legal_moves = environ_state['legal_moves']
        self.turn_index = environ_state['turn_index']
        self.turn_list = environ_state['turn_list']
        self.curr_turn = self.turn_list[self.turn_index]
        # legal_moves_at_valid_games is a dataframe with many rows, and one col, which is the curr turn
        self.legal_moves_at_valid_games = self.get_legal_moves_at_valid_games()
        # self.sorted_legal_moves_at_valid_games = self.legal_moves_at_valid_games.sort_values(by = [self.curr_turn + '-Numeric'], ascending = False)
        
        if self.is_trained: 
            return self.policy_game_mode()
        else:  # the agent is not trained OR this agent instance is the opposing agent

            # it's possible that there will be a completely new set of legal_moves that is not represented in the chess database
            # in that case, legal_moves_at_valid_games will be an empty dataframe, 
            # when that happens, update the Q_table and then pick the first entry in self.legal_moves list and return it to the caller
            if (self.legal_moves_at_valid_games.shape[0] == 0):
                self.update_Q_table()
                self.curr_game = self.chess_data.sample().index.values[0]  # reset the curr game to random
                return {'chess_move_str': self.legal_moves[0], 'chess_move_val': self.settings.new_move_pts, 'move_source': 'unique set of legal moves! q table updated'}
            
            # if the curr move in the sequence is playable... this almost never happens in actual play though
            elif self.is_this_move_legal():  
                chess_move_str = self.chess_data.at[self.curr_game, self.curr_turn]
                chess_move_val = self.chess_data.at[self.curr_game, self.curr_turn + '-Numeric']
                return {'chess_move_str': chess_move_str, 'chess_move_val': chess_move_val, 'move_source': 'that was the very next move in the curr_game sequence'}

            # if the curr move in the sequence was NOT playable, begin search for playable move
            else: 
                return self.policy_training_mode()
            
    # @timeit
    def policy_game_mode(self):
        """ policy to use during game between human player and agent 
            the agent searches its q table to find the moves with the highest q values at each turn.
            if the chess move is not found in the Q_table at that turn, scan the table index and find the move, 
            then increment the cell at this move and turn. if the move is not found in the index, then modify 
            the q_table with this new unique move and increment the q_table at the corresponding cell.
            :param none
            :return a dictionary with chess_move as a string 
        """
        q_table_move_str_list = []
        # q_table_move_list is a pd Series of chess moves sorted in desc order by chess_move_val
        q_table_move_list = self.get_Q_values()
        
        # q_table_move_list.index[0] would be the move with the highest val for a given turn in the Q_table
        # loop through q_table_move_list until a playable move is found
        
        if q_table_move_list.index.any() in self.legal_moves:
            
            matching_moves = q_table_move_list.filter(items = self.legal_moves) # matching_moves is a pandas series
            for chess_move_str in matching_moves.index:
                # load up a list of the moves we can play (if any)
                q_table_move_str_list.append(chess_move_str)
        
            dice_roll_random = math.floor(random.uniform(0, 1 / (1 - self.settings.chance_for_random))) 
            if dice_roll_random == 1:
                
                chess_move_str = random.choice(q_table_move_str_list)  # get a random chess move
                if self.Q_table.at[chess_move_str, self.curr_turn] == 0:
                    self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)
                
                return {'chess_move_str': chess_move_str, 'move_source': 'move was already in the Q_table at curr_turn, randomly chosen'}
            
            else: # return the first available legal move

                chess_move_str = q_table_move_str_list[0]
                if self.Q_table.at[chess_move_str, self.curr_turn] == 0:
                    self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)
                
                return {'chess_move_str': chess_move_str, 'move_source': 'move was already in the Q_table at curr_turn'}

        # if we reach this point in the code, that means the q_table does not have a valid move at all,
        # which means that list of unique moves has increased (should be very rare)
        # add these unique moves to the Q_table
        else:
            self.update_Q_table()         
            chess_move_str = self.legal_moves[0] # arbitrarily pick a chess_move from the list of legal moves

            # increment q table for new selected chess move
            self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)
            return {'chess_move_str': chess_move_str, 'move_source': 'unique set of legal moves! q table updated'}
    
    
    # @timeit
    def policy_training_mode(self):
        """ this policy determines how the agents choose a move at each turn during training
            :param none
            :return dictionary with selected chess move information
        """
        # dice_roll_random = math.floor(random.uniform(0, 1 / (1 - self.settings.chance_for_random))) 

        ########## prioritize important moves ##########
        # if important move is found, return to caller
        valuable_move = self.choose_high_val_move()  # this method returns 0 when the move is not a special move
        if valuable_move != 0:  # move is a special move, return it
            return valuable_move # this is a dictionary, same form as the return at the end of this method

        # there were no important moves this turn so choose a random move, or a move with the highest value.
        else:
            # pick random move, from a list of available moves. 
            self.curr_game = self.legal_moves_at_valid_games.sample().index.values[0]
            chess_move_str = self.chess_data.at[self.curr_game, self.curr_turn]
            chess_move_val = self.chess_data.at[self.curr_game, self.curr_turn + '-Numeric']
            method_chosen = 'random'
            return {'chess_move_str': chess_move_str, 'chess_move_val': chess_move_val, 'move_source': method_chosen}

        # else: 
        #     self.curr_game = self.sorted_legal_moves_at_valid_games.index[0]
        #     chess_move_str = self.sorted_legal_moves_at_valid_games.iloc[0, 0]
        #     chess_move_val = self.sorted_legal_moves_at_valid_games.iloc[0, 1]
        #     method_chosen = 'greedy'

        
    def choose_high_val_move(self):
        """ method is used during training mode. during training, the dataframe of games with legal moves
            at current turn will be scanned for any strings that match the following regex options.
            the game at which the move occurs will be noted, and the chess move infor will be returned to caller. 
            :param none
            :return selected chess move and other information 
        """ 
        max_val = 0
        # we need to go through the entire list and pick the best move (or zero is there is no good move)
        for game_num in self.legal_moves_at_valid_games.index:
            # we can simply refer to the main chess_data
            chess_move_str = self.chess_data.at[game_num, self.curr_turn]

            # checkmate will always be the best move, so return immediately
            if re.search(r'\#', chess_move_str):
                self.curr_game = game_num
                chess_move_val = self.chess_data.at[game_num, self.curr_turn + '-Numeric']
                method_chosen = 'checkmate'
                return {'chess_move_str': chess_move_str, 'chess_move_val': chess_move_val, 'move_source': method_chosen}

            if re.search(r'=Q', chess_move_str): 
                Q_move_val = 50 # assigned arbitrarily
                if Q_move_val > max_val:
                    max_val = Q_move_val
                    promo_Q_chess_move_str = copy.copy(chess_move_str)
                    self.curr_game = game_num
                    promo_Q_chess_move_val = copy.copy(self.chess_data.at[game_num, self.curr_turn + '-Numeric'])
                    promo_Q_method_chosen = 'promotion to queen'
                    promotion_to_queen = {'chess_move_str': promo_Q_chess_move_str, 'chess_move_val': promo_Q_chess_move_val, 'move_source': promo_Q_method_chosen}

            if re.search(r'=', chess_move_str): 
                promotion_move_val = 25   
                if promotion_move_val > max_val:
                    max_val = promotion_move_val
                    self.curr_game = game_num
                    promo_chess_move_str = copy.copy(chess_move_str)
                    promo_chess_move_val = copy.copy(self.chess_data.at[game_num, self.curr_turn + '-Numeric'])
                    promo_method_chosen = 'promotion'
                    promotion = {'chess_move_str': promo_chess_move_str, 'chess_move_val': promo_chess_move_val, 'move_source':  promo_method_chosen}
                
            if re.search(r'\+', chess_move_str):
                check_move_val = 10
                if check_move_val > max_val:
                    max_val = check_move_val
                    self.curr_game = game_num
                    check_chess_move_str = copy.copy(chess_move_str)
                    check_chess_move_val = copy.copy(self.chess_data.at[game_num, self.curr_turn + '-Numeric'])
                    check_method_chosen = 'check'
                    check_move = {'chess_move_str': check_chess_move_str, 'chess_move_val': check_chess_move_val, 'move_source': check_method_chosen}

            if re.search(r'x', chess_move_str):
                capture_move_val = 5 
                if capture_move_val > max_val:
                    max_val = capture_move_val
                    self.curr_game = game_num
                    capture_move_str = copy.copy(chess_move_str)
                    capture_chess_move_val = copy.copy(self.chess_data.at[game_num, self.curr_turn + '-Numeric'])
                    capture_method_chosen = 'capture'
                    capture_move = {'chess_move_str': capture_move_str, 'chess_move_val': capture_chess_move_val, 'move_source': capture_method_chosen}
            
        if max_val == 50:
            return promotion_to_queen
        elif max_val == 25:
            return promotion
        elif max_val == 10:
            return check_move
        elif max_val == 5:
            return capture_move
        else: 
            return 0
    # end of choose_high_val_move
            

    # @timeit
    def is_this_move_legal(self):
        """ checks a chess_move string at the current turn for a given game sequence, 
            and determines if it is a playable move 
            :param environ_state
            :return bool, True if the move is legal
        """
        # if this move is in the series of moves, across all games with legal moves
        if self.chess_data.loc[self.curr_game, self.curr_turn] in self.legal_moves_at_valid_games[self.curr_turn]:
            return True
        else:
            return False


    # @timeit   
    def get_legal_moves_at_valid_games(self):
        """ function will find the pd Series of playable moves for a given state. Index of this Series is the game_nums
            the col values are the actual moves

            :param environ_state, the state which is a dict containing the gturn, and the legal moves list at that point in the game.
            :return pandas dataframe where the rows are the games that have a legal move at the curr turn. The columns are the chess moves at curr turn, 
             and the numeric value of the chess moves. 
        """
        # this assigns a dataframe to valid_games, each rows (a game sequence) has a valid move at the curr turn
        valid_games = self.chess_data[self.chess_data.loc[:, self.curr_turn].isin(self.legal_moves)]
        legal_moves_df = valid_games.loc[:, [self.curr_turn, self.curr_turn + '-Numeric']]
        return legal_moves_df.drop_duplicates(subset = self.curr_turn)
    
    
    # @timeit
    def init_Q_table(self, chess_data):
        """ creates the q table so the agent can be trained 
            the q table index represents unique moves across all games in the database for all turns.
            columns are the turns, 'W1' to 'B75'
            :param none
            :return q_table a pandas dataframe
        """
        # initialize array that will be used to build a list of pd Series.
        # each Series represents the unique moves for the turn for a player color, W1 for example.
        uniq_mov_list = []

        # this loop will make an array of pandas series, add 1 to make it 1 through total columns (inclusive)
        for i in range(1, self.settings.num_columns + 1):
            uniq_mov_list.append(chess_data.loc[:, self.color + str(i)].value_counts())

        # make a big series by concatenating all series.
        uniq_mov_list = pd.concat(uniq_mov_list)
        
        # drop duplicate rows indices and return the row index, this will be used to make a pd df
        uniq_mov_list = uniq_mov_list.index.drop_duplicates(keep = 'first')

        # make a list of columns that represents the player turns, W1, W2, ... Wn for example.
        turns_list = chess_data.loc[:, self.color + '1': self.color + str(self.settings.num_columns): 2].columns

        # use turn list and uniq move list to make df, fill with 0s and change dtype
        q_table = pd.DataFrame(columns = turns_list, index = uniq_mov_list)

        # make all colum values in df 0.
        for col in q_table.columns:
            q_table[col].values[:] = 0

        q_table = q_table.astype(np.int32)
        return q_table # returns a pd dataframe


    def change_Q_table_pts(self, chess_move, curr_turn, pts):
        """ 
            :param chess_move is a string, 'e4' for example
            :param curr_turn is a string representing the turn num, for example, 'W10'
            :param pts is an int for the number of points to add to a q table cell
            :return none
        """
        self.Q_table.at[chess_move, curr_turn] += pts


    # @timeit
    def update_Q_table(self):
        q_table_new_values = pd.DataFrame(index = self.legal_moves, columns = self.Q_table.columns, dtype = np.int32)

        # zero out new entries in the q_table.
        for col in q_table_new_values.columns: 
            q_table_new_values[col].values[:] = 0

        self.Q_table = pd.concat([self.Q_table, q_table_new_values])
        # protect against duplicate indices, shouldnt happen, but it might
        self.Q_table = self.Q_table[~self.Q_table.index.duplicated()]
        
        
    def reset_Q_table(self):
        """ zeroes out the q table, call this method when you want to retrain the agent """
        for col in self.Q_table.columns:
            self.Q_table[col].values[:] = 0


    def get_Q_values(self):
        """ 
            Returns the series for the given turn, sorted by value in desc order. the series represents the moves for that turn.
            The returned series represents the best action by the agent based on numerical value.
            the agent can use this series at each turn to make a move, as long as it's a legal move.
            :param curr_turn is a string, like 'W10'
            :return a pandas series, the index represents the chess moves, and the col is the curr turn in the game.
        """
        return self.Q_table.loc[:, self.curr_turn].sort_values(ascending = False)
    

    # # returns the first n rows of the q table
    # def peek_Q_table(self, num_rows):
    #     """ returns the first n rows in the q table dataframe """
    #     return self.Q_table.head(num_rows)

