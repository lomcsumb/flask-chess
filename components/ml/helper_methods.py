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
            # print(f"legal moves are: {bubs.environ.legal_moves}\n")
            chess_move = bubs.rl_agent_chess_move() # chess_move is a dictionary
            chess_move_str = chess_move['chess_move_str']
            chess_move_src = chess_move['move_source']
            print(f'RL agent played {chess_move_str} - from source: {chess_move_src}\n')

        else:
            print('=== OPPONENT\' TURN ===')
            # print(f"legal moves are: {bubs.environ.legal_moves}\n")
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


# ============== MOVE ANALYSIS WITH STOCKFISH

# def init_stockfish_engine(stockfish_filepath):
#     """ creates an engine object that will be used for fen analysis """
#     engine = chess.engine.SimpleEngine.popen_uci(stockfish_filepath)
#     return engine


# def get_move_value_sf(fen_str, stockfish_filepath, num_moves_to_return = 1, depth_limit = 4, time_limit = None):
#     """ this function will return a move score based on the analysis results from stockfish 
#         :params standard stockfish settings
#         :return an int, representing the value of a chess move, based on the fen score.
#     """
#     engine = chess.engine.SimpleEngine.popen_uci(stockfish_filepath)

#     search_limit = chess.engine.Limit(depth = depth_limit, time = time_limit)
#     board = chess.Board(fen_str)
#     infos = engine.analyse(board, search_limit, multipv = num_moves_to_return)
#     move_score = [format_info(info) for info in infos][0]
#     move_score = (move_score['mate_score'], move_score['centipawn_score'])
#     if (move_score[1] is None): # centipawn score is none, that means there is a mate_score, choose that
#         # mate_score is arbitrarily set high, much higher than a pawn score would ever be
#         # this is the assigned centipawn score, 10,000 score would never happen, 
#         # but something like 4_000 can sometimes happen
#         move_score = 10_000  
#     else:
#         # select the centipawn score, most of the time this will be < 100, but it may also get into the thousands
#         move_score = move_score[1]
    
#     return move_score


# def format_info(info):
#     # Normalize by always looking from White's perspective
#     score = info["score"].white()
    
#     # Split up the score into a mate score and a centipawn score
#     mate_score = score.mate()
#     centipawn_score = score.score()
#     return {
#         "mate_score": mate_score,
#         "centipawn_score": centipawn_score,
#         "pv": format_moves(info["pv"]),
#     }


# # Convert the move class to a standard string 
# def format_moves(pv):
#     return [move.uci() for move in pv]


# ===================== TIMEIT ===============================
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
