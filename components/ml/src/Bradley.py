from components.ml.src.Environ import Environ
from components.ml.src.Agent import Agent
from components.ml.src.Settings import Settings
import pandas as pd
import numpy as np
import re
import chess
import chess.engine
from datetime import date
from components.ml.src.helper_methods import *

class Bradley:
    """ 
        Bradley class acts as the single point of communication between the RL agent and the player.
        this class trains the agent and helps to manage the chessboard during play between the comp and the
        user. This is a composite class with members of the Environ class and the Agent class
    """
    def __init__(self, chess_data, rl_agent_color = 'W'):    
        self.chess_data = chess_data
        self.settings = Settings()
        self.environ = Environ(self.chess_data)
        
        # init agent that we will train
        rand_game = self.get_rand_game()
        self.rl_agent = Agent(rl_agent_color, self.chess_data, rand_game)
        
        # use this opponent agent to play against rl_agent
        opp_agent_color = self.get_opp_agent_color(rl_agent_color)
        is_opp_agent = True
        rand_game = self.get_rand_game()
        self.opp_simple_agent = Agent(opp_agent_color, self.chess_data, rand_game, is_opp_agent)

        # stockfish will be used to assign points to chess moves
        # self.engine = chess.engine.SimpleEngine.popen_uci(self.settings.stockfish_filepath)
        self.num_moves_to_return = 1
        self.depth_limit = 4
        self.time_limit = None
        self.search_limit = chess.engine.Limit(depth = self.depth_limit, time = self.time_limit)

    # @timeit
    def recv_opp_move(self, chess_move):                                                                                 
        """ receives opp chess move and send it to the environ so that the chessboard can be loaded w/ opponent move.
            this method assumes that the incoming chess move is valid and playable.
            :param chess_move this is the str input, which comes from the human player
            :return bool, True for successful move input
        """
        # load_chessboard returns False if failure to add move to board,
        if self.environ.load_chessboard(chess_move):
            # loading the chessboard was a success, now just update the curr state
            self.environ.update_curr_state()
            return True
        else:
            return False
            
    # @timeit
    def rl_agent_chess_move(self):
        """ this method will do two things, load up the environ.chessboard with the agent's action (chess_move)
            and return the string of the agent's chess move
            We can also use this method to play against a simple agent that is not trained.

            :param none
            :return a dictionary with the chess move str as one of the items
        """
        curr_state = self.environ.get_curr_state()
        chess_move = self.rl_agent.choose_action(curr_state)
        self.environ.load_chessboard(chess_move['chess_move_str'])
        self.environ.update_curr_state()
        return chess_move  # this is a dictionary


    def get_fen_str(self):
        """ call this at each point in the chess game for the fen 
            string. The FEN string can be used to reconstruct a chessboard.
            The FEN string will change each time a move is made.
            :param none
            :return a string that represents a board state, something like this,
            'rnbqkbnr/pppp1ppp/8/8/4p1P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3'
        """
        return self.environ.board.fen()


    def get_opp_agent_color(self, rl_agent_color):
        """ method determines what color the opposing rl agent should be assigned to
            :param string that represents the rl_agent color 
            :return string that represents the opposing rl agent color
        """
        if rl_agent_color == 'W':
            return 'B'
        else:
            return 'W'
    
    # @timeit
    def get_rand_game(self):
        """ returns a random game number from the chess_data pandas dataframe 
            :param none
            :return a single string val, 'Game 2' for example
        """
        return self.chess_data.sample().index.values[0]
    
    
    def get_curr_turn(self):
        """ find curr turn which would be a string like 'W1' or 'B5'
            :param none
            :return a string representing the current turn
        """
        return self.environ.get_curr_turn()
    
    # @timeit
    def game_on(self):
        """ method determines when the game is over 
            game can end if, python chess board determines the game is over, 
            or if the game is at 75 moves per player
            :param none
            :return bool, False means the game is over
        """
        if self.environ.board.is_game_over() or self.environ.turn_index >= 149:
            return False
        else:
            return True
    
    
    def get_legal_moves(self):
        """ returns a list of strings that represents the legal moves for the 
            current turn and state of the chessboard
            :param none
            :return a list of strings
        """
        return self.environ.legal_moves
    
    
    def get_rl_agent_color(self): 
        """ returns the string for the color of the agent
            :param none
            :return string, 'W' for example
        """
        return self.rl_agent.color
    
    
    def get_game_outcome(self):   
        """ method will return the winner, loser, or draw for a chess game
            :param none
            :return chess.Outcome class, this class has a result() method that returns 1-0, 0-1 or 1/2-1/2 
                or false if failure
        """
        try:
            return self.environ.board.outcome().result()
        except AttributeError:
            return 'outcome not available, most likely game ended because turn_index was too high'
    
    
    def get_game_termination_reason(self):
        """ the method determines why the game ended, for example
            if the game ended due to a checkmate, a string termination.CHECKMATE will be returned
            :param none
            :return a single string that describes the reason for the game ending
        """
        try:
            return str(self.environ.board.outcome().termination)
        except AttributeError:
            return 'termination reason not available, most likely game ended because turn_index was too high'

    
    def get_chessboard(self):
        """ returns the chessboard object
            the chessboard object can be printed and the output will be an ASCII representation of the chessboard
            and current state of the game.
            :param none
            :return chessboard object
        """
        return self.environ.board

    
    # @timeit
    def train_rl_agent(self, training_file_path, num_games):
        """ trains the agent and then sets is_trained flag to True.
            the algorithm used for training is SARSA. an opposing agent object
            is utililzed to train the main rl_agent that will play against the human player
            A chess game can end at multiple places during training, so we need to 
            check for end-game conditions throughout this method.
            It's possible to train a new agent with an already trained agent, 
            just reassign like this, untrained_agent.opp_simple_agent = trained_agent.rl_agent
            and then call this method.

            :param training_file_path, where to write the output of training data
            :param num_games, how long to train the agent
            :return none
        """ 
        training_results = open(training_file_path, 'a')
        today = date.today()
        training_results.write("\n\n\n========== START OF TRAINING PHASE ==========\n\n")
        training_results.write(f"Date: {today}")
        training_results.write(f"\nAgent settings are, learn rate: {self.rl_agent.settings.learn_rate}, discount factor: {self.rl_agent.settings.discount_factor}\n")
           
        curr_QVal = self.settings.initial_q_val
        
        for curr_training_game in range(num_games):
            #initialize environment to provide a state, s
            curr_state = self.environ.get_curr_state()
            
            training_results.write(f"\n\n *** Start of game {curr_training_game + 1} ***\n")
            training_results.write('chessboard is : \n')
            training_results.write(str(self.get_chessboard()))
            training_results.write('\n\n')
            
            if self.opp_simple_agent.color == 'W':
                training_results.write('\nopposing agent is playing as White\n')
                training_results.write(f"current turn is: {self.environ.get_curr_turn()}\n")
                training_results.write(f"turn_index = {curr_state['turn_index']}\n")
                training_results.write(f"legal_moves are:\n {curr_state['legal_moves']}\n")
                
                # opponent chooses move
                opp_curr_action = self.opp_simple_agent.choose_action(curr_state)

                training_results.write(f"\nopp move is {opp_curr_action['chess_move_str']}\n")
                training_results.write(f"from source: {opp_curr_action['move_source']}\n")
                training_results.write('\n\n')

                # opp plays move, chessboard is loaded
                self.environ.load_chessboard(opp_curr_action['chess_move_str'])              
                self.environ.update_curr_state()
                curr_state = self.environ.get_curr_state()
            
            while self.game_on():     # this check is unneccessary, but I'm keeping it anyway
                ##### RL AGENT PICKS AND PLAYS MOVE #####
                # choose action a from state s, using policy
                curr_action = self.rl_agent.choose_action(curr_state)
                chess_move = curr_action['chess_move_str']
                curr_turn = self.environ.get_curr_turn()
                
                training_results.write("=== RL AGENT'S TURN ===\n")
                training_results.write(f"current turn is: {self.environ.get_curr_turn()}\n")
                training_results.write(f"turn_index = {curr_state['turn_index']}\n")
                training_results.write(f"legal_moves = {curr_state['legal_moves']}\n")

                # add to Q table
                pts = curr_QVal
                training_results.write('\nadding points to the q table: \n')
                training_results.write(f'{pts} points\n')
                training_results.write(f"at chess move: {chess_move}\n")
                training_results.write(f'and curr turn: {curr_turn}\n')
                
                self.rl_agent.change_Q_table_pts(chess_move, curr_turn, pts)
                
                # take action a, observe r, s', and load chessboard
                self.environ.load_chessboard(chess_move)
                self.environ.update_curr_state()
                curr_state = self.environ.get_curr_state()
                reward = self.get_reward(chess_move)

                training_results.write(f"\nrl agent move is {curr_action['chess_move_str']}\n")
                training_results.write(f"from source: {curr_action['move_source']}\n")
                training_results.write(f"reward is : {reward}\n\n")


                ##### OPPONENT AGENT PICKS AND PLAYS MOVE #####
                if self.game_on():
                    training_results.write("=== OPPONENT'S TURN ===\n")
                    training_results.write(f"current turn is: {self.environ.get_curr_turn()}\n")
                    training_results.write(f"turn_index = {curr_state['turn_index']}\n")
                    training_results.write(f"legal_moves = {curr_state['legal_moves']}\n")

                    # opponent chooses move
                    opp_curr_action = self.opp_simple_agent.choose_action(curr_state)

                    training_results.write(f"\nopp move is {opp_curr_action['chess_move_str']}\n")
                    training_results.write(f"from source: {opp_curr_action['move_source']}\n\n")

                    # opp plays move, chessboard is loaded
                    self.environ.load_chessboard(opp_curr_action['chess_move_str'])
                    self.environ.update_curr_state()
                    curr_state = self.environ.get_curr_state()

                else:
                    # game ended after rl agent's chess move
                    training_results.write(f'\n\n\nGame is over\n') 
                    training_results.write(f'Game result is: {self.get_game_outcome()}\n')
                    training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')

                    training_results.write(f'turn index is: {self.environ.turn_index}\n')
                    training_results.write(f'end of Game: {curr_training_game + 1} training loop\n')
                    training_results.write('Chessboard is: \n')
                    training_results.write(str(self.get_chessboard()))
                    training_results.write('\n\n')

                    self.reset_environ()
                    break # break out of this game and go to next training game

                ##### RL AGENT CHOOSES ACTION, BUT DOES NOT PLAY IT!!! #####
                # observe next_state, s' (this would be after the opponent moves)
                # and choose action a' ( choose, but don't play it! this step is very important)
                if self.game_on():
                    # next_state ... the next state is what the player wants the chessboard
                    # to look like, so that the player can stick to his plan (the curr game sequence)
                    # the next action is the next move in the current game sequence
                    next_action = self.rl_agent.choose_next_action(curr_state)
                    est_QVal = next_action['chess_move_val']

                    training_results.write('this is the PROPOSED chess move by the agent for next turn\n')
                    training_results.write(f"rl agent move choice is {next_action['chess_move_str']}\n")
                    training_results.write('\n')

                else: # game ended after opposing agent's chess move
                    training_results.write(f'\nGame is over\n') 
                    training_results.write(f'Game result is: {self.get_game_outcome()}\n')
                    training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')

                    training_results.write(f'turn index is: {self.environ.turn_index}\n')
                    training_results.write(f'end of Game: {curr_training_game + 1} training loop\n')
                    training_results.write('chessboard looks like this: \n')
                    training_results.write(str(self.get_chessboard()))
                    training_results.write('\n')

                    self.reset_environ()
                    break # break out of this game and go to next training game

                # CRITICAL STEP
                next_QVal = curr_QVal + self.rl_agent.settings.learn_rate * (reward + ((self.rl_agent.settings.discount_factor * est_QVal) - curr_QVal))
                
                training_results.write(f"curr_QVal is: {curr_QVal}\n")
                training_results.write(f"next_QVal is: {next_QVal}\n\n")

                chess_move = curr_action['chess_move_str']
                pts = next_QVal
                curr_turn = self.environ.get_curr_turn()

                training_results.write('adding points to the q table: \n')
                training_results.write(f'{pts} points\n')
                training_results.write(f"at chess move: {chess_move}\n")
                training_results.write(f'and curr turn: {curr_turn}\n')

                # on the first turn, this is assigning points to col W2 in the Q table (assuming rl_agent plays as W) and so on
                self.rl_agent.change_Q_table_pts(chess_move, curr_turn, pts)
                           
                curr_QVal = next_QVal

                training_results.write('\nend of this round, chessboard looks like this: \n')
                training_results.write(str(self.get_chessboard()))
                training_results.write('\n\n\n')
                ### end of curr game round , each round is something like W10 then B10, so basically each time both players have a turn ###
            
            # reset environ to prepare for the next game
            self.reset_environ()
        
        training_results.write(f'Training is complete\n')
        training_results.write(f'Agent was trained on {curr_training_game + 1} Games total\n')
        self.rl_agent.is_trained = True
        self.reset_environ()


    @timeit
    # add this param when you want to print to a file, calculate_pts_filepath
    def calculate_move_pts(self):        
        """ This method is used to play through the games in the database exactly as shown.
            One agent plays White's moves exactly, and Black does the same.
            The objective is to get the point value of a position at each turn for both players.
            Stockfish is used to analyze each position and return an int that represents centipawn score
            This number represents a rudimentary analysis of who is currently winning, but at each turn 
            centipawn scores can change dramatically, so it's not a long term predictor.
        """
        # calculate_pts_report = open(calculate_pts_filepath, 'a')
        self.reset_environ()
        
        # iter through games, at each game, the two agents play out the game exactly as it's shown in the df
        for game_num in self.chess_data.index:
            num_chess_moves = self.chess_data.loc[game_num, 'Num Moves']
            # calculate_pts_report.write(f"\n========== GAME {game_num} ANALYSIS & POINT ASSIGNMENT ==========\n")
            # calculate_pts_report.write(f'Starting FEN string is: {self.environ.board.fen()}\n\n')
            # calculate_pts_report.write(f"\n=== this game loop will iterate {num_chess_moves} times ===\n")

            while self.environ.turn_index < num_chess_moves:
                ##### 'W' PICKS AND PLAYS MOVE #####
                curr_state = self.environ.get_curr_state()
                curr_action = self.rl_agent.choose_action_move_pts_calc(curr_state)
                chess_move = curr_action['chess_move_str']
                curr_turn = self.environ.get_curr_turn()
                
                # calculate_pts_report.write("=== W's turn ===\n")
                # calculate_pts_report.write(f"current turn is: {curr_turn}\n")
                # calculate_pts_report.write(f"turn_index = {curr_state['turn_index']}\n")

                self.environ.load_chessboard(chess_move)
                W_pts = self.get_move_value_sf() # stockfish analyzes the board, to return an int

                # now add insert those points into the chess dataframe at turn_index + -numeric, and curr game
                self.chess_data.at[game_num, curr_turn + '-Numeric'] = W_pts
                # calculate_pts_report.write(f"\nW agent move is {chess_move}\n")
                # calculate_pts_report.write(f'the curr move gives {W_pts} points\n')
                # calculate_pts_report.write(f'FEN string is: {self.environ.board.fen()}\n\n')

                # calculate_pts_report.write(f'the chess db cell at {game_num} game and {curr_turn}-Numeric is assigned {W_pts} points\n\n')
                
                self.environ.update_curr_state()
                
                ##### 'B' PICKS AND PLAYS MOVE #####
                # sometimes the Black resigns this would cause an out of index problem here. 
                # so check for it. If White resigns, that's not a problem.
                curr_state = self.environ.get_curr_state()

                if curr_state['turn_index'] >= num_chess_moves:
                    # calculate_pts_report.write(f'\nGame is over\n') 
                    # calculate_pts_report.write(f'turn index is: {self.environ.turn_index}\n\n')
                    # calculate_pts_report.write(f'===== End of {game_num} game analysis =====\n\n')
                    # calculate_pts_report.write('Chessboard is: \n')
                    # calculate_pts_report.write(str(self.get_chessboard()))
                    # calculate_pts_report.write('\n\n')
                    # calculate_pts_report.write(f'FEN string is: {self.environ.board.fen()}\n\n')
                    # calculate_pts_report.write('\n\n\n')
                    self.reset_environ()
                    break # game over, go to next game
                
                opp_curr_action = self.opp_simple_agent.choose_action_move_pts_calc(curr_state)
                chess_move = opp_curr_action['chess_move_str']
                curr_turn = self.environ.get_curr_turn()

                # calculate_pts_report.write("=== B's TURN ===\n")
                # calculate_pts_report.write(f"current turn is: {self.environ.get_curr_turn()}\n")
                # calculate_pts_report.write(f"turn_index = {curr_state['turn_index']}\n")                               

                # opp plays move, chessboard is loaded
                self.environ.load_chessboard(chess_move)
                B_pts = self.get_move_value_sf() # stockfish analyzes the board, to return an int
                
                # calculate_pts_report.write(f"\nB agent move is {chess_move}\n")
                # calculate_pts_report.write(f'the curr move gives {B_pts} points\n')
                # calculate_pts_report.write(f'FEN string is: {self.environ.board.fen()}\n\n')
                # calculate_pts_report.write(f'the chess db cell at {game_num} game and {curr_turn}-Numeric would have been assigned {B_pts} points\n\n')

                self.environ.update_curr_state()

                # calculate_pts_report.write('end of this round, chessboard looks like this: \n')
                # calculate_pts_report.write(str(self.get_chessboard()))
                # calculate_pts_report.write('\n\n\n')
                
            # reset environ to prepare for the next game
            # calculate_pts_report.write(f'\nGame is over\n') 
            # calculate_pts_report.write(f'turn index is: {self.environ.turn_index}\n\n')
            # calculate_pts_report.write(f'===== End of {game_num} game analysis =====\n\n')
            # calculate_pts_report.write('Chessboard is: \n')
            # calculate_pts_report.write(str(self.get_chessboard()))
            # calculate_pts_report.write('\n\n')
            # calculate_pts_report.write(f'FEN string is: {self.environ.board.fen()}\n\n')
            # calculate_pts_report.write('\n\n\n')
            self.reset_environ()

        # calculate_pts_report.write(f'chess move pts assignment is done.\n')
        # calculate_pts_report.write('points for db moves are: \n\n')
        # calculate_pts_report.write(self.chess_data.loc[:, 'W1-Numeric': 'W75-Numeric'].to_string())
        # calculate_pts_report.write('\n\ncalculate_move_pts method is done')
        # calculate_pts_report.close()
        self.reset_environ()
        

    def get_move_value_sf(self):
        """ this function will return a move score based on the analysis results from stockfish 
            :params standard stockfish settings
            :return an int, representing the value of a chess move, based on the fen score.
        """
        infos = self.engine.analyse(self.environ.board, self.search_limit, multipv = self.num_moves_to_return)
        move_score = [self.format_info(info) for info in infos][0]
        move_score = (move_score['mate_score'], move_score['centipawn_score'])

        if (move_score[1] is None): # centipawn score is none, that means there is a mate_score, choose that
            # mate_score is arbitrarily set high, much higher than a pawn score would ever be
            # this is the assigned centipawn score, 10,000 score would never happen, 
            # but something like 4_000 can sometimes happen
            return 10_000  
        else:
            # select the centipawn score, most of the time this will be < 100, but it may also get into the thousands
            return move_score[1]


    def format_info(self, info):
        # Normalize by always looking from White's perspective
        score = info["score"].white()

        # Split up the score into a mate score and a centipawn score
        mate_score = score.mate()
        centipawn_score = score.score()
        return {
            "mate_score": mate_score,
            "centipawn_score": centipawn_score,
            "pv": self.format_moves(info["pv"]),
        }


    # Convert the move class to a standard string 
    def format_moves(self, pv):
        return [move.uci() for move in pv]

        
    def get_reward(self, chess_move_str):                                     
        """
            returns the number of points for a special chess action
            :param chess_move, string representing selected chess move
            :return reward based on type of move
        """
        total_reward = 0
        if re.search(r'N.', chess_move_str): # encourage development of pieces
            total_reward += 1_000
        if re.search(r'R.', chess_move_str):
            total_reward += 1_000
        if re.search(r'B.', chess_move_str):
            total_reward += 1_000
        if re.search(r'Q.', chess_move_str):
            total_reward += 1_000
        if re.search(r'x', chess_move_str):    # capture
            total_reward += 5_000
        if re.search(r'\+', chess_move_str):   # check
            total_reward += 10_000
        if re.search(r'=', chess_move_str):
            total_reward += 20_000 # promotion
        if re.search(r'=Q', chess_move_str):    # a promotion to Q
            total_reward += 50_000
        if re.search(r'\#', chess_move_str): # checkmate
            total_reward += 100_000

        return total_reward

        
    def reset_environ(self):
        self.environ.reset_environ()