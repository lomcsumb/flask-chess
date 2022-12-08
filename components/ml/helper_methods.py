#####
# import Bradley as imman
import components.ml.Bradley as imman
import pandas as pd
from functools import wraps
import time
## Hello World

def init_agent(chess_data, color = 'W'):
    bubs = imman.Bradley(chess_data, color)
    return bubs


def play_game(bubs):
    W_turn = True
    turn_num = bubs.get_curr_turn()
    
    while bubs.game_on():
        if W_turn:
            player_turn = 'W'
        else:
            player_turn = 'B'
        
        print(f'\nCurrent turn is :  {turn_num}')
        
        if bubs.get_rl_agent_color() == player_turn:
            print('=== RL AGENT\'S TURN ===')
            chess_move = bubs.rl_agent_chess_move()
            chess_move_str = chess_move['chess_move_str']
            print(f'RL agent played {chess_move_str}\n')
        else:
            print('=== OPPONENT\' TURN ===')
            chess_move = str(input('hooman, enter chess move: '))
            print('\n')
            
            while not bubs.recv_opp_move(chess_move):  # this method returns False for incorrect input
                print('invalid input, try again')
                chess_move = str(input('enter chess move: '))
        
        turn_num = bubs.get_curr_turn()
        W_turn = not W_turn # simple flag to switch the turn to B or vice-versa
    
    print(f'Game is over, result is: {bubs.get_game_outcome()}')
    print(f'The game ended because of: {bubs.get_game_termination_reason()}')


def agent_vs_agent(bubs, imman):
    W_turn = True
    turn_num = bubs.get_curr_turn()
    
    while bubs.game_on():
        if W_turn:
            player_turn = 'W'
        else:
            player_turn = 'B'
        
        # bubs's turn
        print(f'\nCurrent turn:  {turn_num}')
        if bubs.get_rl_agent_color() == player_turn:
            chess_move_bubs = bubs.rl_agent_chess_move()
            bubs_chess_move_str = chess_move_bubs['chess_move_str']
            bubs_chess_move_src = chess_move_bubs['move_source']
            print(f'Bubs played {bubs_chess_move_str} - from source: {bubs_chess_move_src}\n')
            print(bubs.environ.board)
            imman.recv_opp_move(bubs_chess_move_str)

        # imman's turn
        else:
            chess_move_imman = imman.rl_agent_chess_move()
            imman_chess_move_str = chess_move_imman['chess_move_str']
            imman_chess_move_src = chess_move_imman['move_source']
            print(f'Imman played {imman_chess_move_str} - from source: {imman_chess_move_src}\n')
            print(imman.environ.board)
            bubs.recv_opp_move(imman_chess_move_str)
        
        turn_num = bubs.get_curr_turn()
        W_turn = not W_turn # simple flag to switch the turn to B or vice-versa
    
    print(bubs.environ.board)
    print('\n\n')
    print(imman.environ.board)
    print(f'Game is over, result is: {bubs.get_game_outcome()}')
    print(f'The game ended because of: {bubs.get_game_termination_reason()}')


def pikl_q_table(bubs, q_table_path):
    bubs.rl_agent.Q_table.to_pickle(q_table_path, compression = 'zip')


def pikl_chess_data(bubs, chess_data_filepath):
    bubs.chess_data.to_pickle(chess_data_filepath, compression = 'zip')


def bootstrap_agent(bubs, existing_q_table_path):
    """skip training the agent by assigning its q table to an existing q table.
        make sure the q table you pass matches the color of the agent you are going
        to play as. 
    """
    bubs.rl_agent.Q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    bubs.rl_agent.is_trained = True
