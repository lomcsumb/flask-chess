# # this is where the agent will be trained. 
# import Bradley as imman
# import pandas as pd
# import numpy as np
# from helper_methods import *

# stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"

# W_q_table_path_100_games_1_000_game_sample = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\W_100_games_1000_game_sample.pkl"
# W_training_file_path_100_games = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\W_100_games_1000_game_sample.txt'
# training_length_100_games = 100

# W_q_table_path_1_000_games_10_000_game_sample = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\W_1000_games_10000_game_sample.pkl"
# W_training_file_path_1_000_games = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\W_1000_games_10000_game_sample.txt'
# training_length_1_000_games = 1_000

# W_q_table_path_10_000_games_100_000_game_sample = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\W_10000_games_100000_game_sample.pkl"
# W_training_file_path_10_000_games = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\W_10000_games_100000_game_sample.txt'
# training_length_10_000_games = 10_000

# calculate_pts_filepath = r'C:\Users\Abrah\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\calculate_pts_report.txt'

# chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"
# chess_data_path_pts_assigned = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\chess_data\chess_data_pts_assigned.pkl"

# DB_1_000_sample_size = 1_000
# DB_10_000_sample_size = 10_000
# DB_100_000_sample_size = 100_000

# # read pkl file that contains the dataframe.
# big_chess_DB = pd.read_pickle(chess_data_path, compression = 'zip')
# chess_data = big_chess_DB.head(1000)


# if __name__ == '__main__':
#     # ========= init agents ========= #
#     bubs = init_agent(chess_data, 'W') # training on sample of data
#     # bubs.rl_agent.Q_table = bubs.rl_agent.init_Q_table(big_chess_DB) 
#     # eman = init_agent(chess_data, 'B')

#     # ========== fill out move points in db
#     bubs.calculate_move_pts()

#     # this df should have the correct assigned points based stockfish analysis
#     # pikl_chess_data(bubs, chess_data_path_pts_assigned)  

#     # ============= train a new agent =============== #
#     # init q table on full db, 
#     # bubs.rl_agent.Q_table = bubs.rl_agent.init_Q_table(big_chess_DB) 
#     # train only on a sample, not the full DB
#     # bubs.train_rl_agent(W_training_file_path_100_games, training_length_100_games)
#     # pikl_q_table(bubs, W_q_table_path_100_games_1_000_game_sample)

#     # ============ use an existing agent ================= #
#     # bootstrap_agent(bubs, W_q_table_path_100_games_1_000_game_sample)
#     # print(f"agent is trained? {bubs.rl_agent.is_trained}")

#     # play against agent
#     # play_game(bubs)

#     # agent v agent
#     # agent_vs_agent(bubs, eman)
