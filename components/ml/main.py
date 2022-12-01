import Bradley as imman
import pandas as pd
import numpy as np
from helper_methods import *

q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499 - Capstone Project\Q_Tables\White_Q_table_100_games.pkl"
chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499 - Capstone Project\chess_data\kaggle_chess_data.pkl"

# read pkl file that contains the dataframe.
big_chess_DB = pd.read_pickle(chess_data_path, compression = 'zip')
chess_data = big_chess_DB.sample(100_000)

if __name__ == '__main__':
    # create new RL agent
    bubs = init_agent(chess_data)
    
    # load the agent with a previously trained agent's Q table
    # don't train an agent when a user want to play the game, very time-consuming
    bubs.rl_agent.Q_table = pd.read_pickle(q_table_path, compression = 'zip') # pikl files load faster and the formatting is cleaner
    bubs.rl_agent.is_trained = True # set this to trained since we assigned a preexisting Q table to new RL agent

    play_game(bubs)