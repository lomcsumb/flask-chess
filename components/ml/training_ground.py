# import pandas as pd
# from helper_methods import *
# import time
# from functools import wraps

# training_length_100_games = 100
# training_length_1_000_games = 1_000
# training_length_10_000_games = 10_000
# training_length_10_000_games = 100_000

# calculate_pts_filepath_White = r'C:\Users\Abrah\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\calculate_pts_report.txt'

# chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"
# chess_data_path_pts_assigned_White = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\chess_data\chess_data_pts_assigned.pkl"

# bradley_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\bradley_agent_q_table.pkl"
# bradley_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\bradley_training_results.txt'

# immanuel_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\immanuel_agent_q_table.pkl"
# immanuel_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\immanuel_training_results.txt'

# bradley_rl_agent_vs_simple_agent_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\bradley_rl_agent_vs_simple_agent_training_results.txt'
# immanuel_rl_agent_vs_simple_agent_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\immanuel_rl_agent_vs_simple_agent_training_results.txt'

# rl_agent_vs_rl_agent_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Training_results\B_rl_agent_vs_rl_agent_training_results.txt'


# DB_10_000_sample_size = 10_000
# DB_50_000_sample_size = 50_000
# DB_100_000_sample_size = 100_000
# DB_1_000_000_sample_size = 1_000_000

# # read pkl file that contains the dataframe.
# # big_chess_DB = pd.read_pickle(chess_data_path, compression = 'zip')
# # chess_data = big_chess_DB.sample(DB_100_000_sample_size)

# # read chess_data thas has been assigned with correct move values
# chess_data = pd.read_pickle(chess_data_path_pts_assigned_White, compression = 'zip')
# small_chess_data = chess_data.sample(20_000)


# if __name__ == '__main__':
#     # ========= init agent(s) ========= #
#     bradley = init_agent(small_chess_data, 'W') # training on sample of data
#     # immanuel = init_agent(small_chess_data, 'B') # training on sample of data

#     # ============= train a new agent =============== #
#     # bradley.train_rl_agent(bradley_training_results_filepath, 1000)
#     # pikl_q_table(bradley, bradley_agent_q_table_path)
#     # print('training done')

#     # immanuel.train_rl_agent(immanuel_training_results_filepath, training_length_1_000_games)
#     # pikl_q_table(immanuel, immanuel_agent_q_table_path)
#     # print('training done')
   

#     # ============= bootstrap and retrain =============== #
#     # to keep training agent, bootstrap it, and simply run training again.
#     # the is_trained flag needs to be set to False for this to work.
#     # bradley = init_agent(small_chess_data, 'W') # training on sample of data
#     # bootstrap_agent(bradley, bradley_agent_q_table_path)
#     # bradley.rl_agent.is_trained = False
#     # bradley.train_rl_agent(bradley_training_results_filepath, 5000)
#     # pikl_q_table(bradley, bradley_agent_q_table_path)
#     # print('finished training')

#     # immanuel = init_agent(small_chess_data, 'B') # training on sample of data
#     # bootstrap_agent(immanuel, immanuel_agent_q_table_path)
#     # immanuel.rl_agent.is_trained = False
#     # immanuel.train_rl_agent(immanuel_training_results_filepath, training_length_1_000_games)
#     # pikl_q_table(immanuel, immanuel_agent_q_table_path)
#     # print('finished training')



#     # ============= bootstrap and play against simple agent =============== #
#     bootstrap_agent(bradley, bradley_agent_q_table_path)
#     # bootstrap_agent(immanuel, immanuel_agent_q_table_path)

#     bradley.rl_agent_vs_simple_agent(bradley_rl_agent_vs_simple_agent_training_results_filepath, 5000)
#     pikl_q_table(bradley, bradley_agent_q_table_path)
#     print('training complete')

#     # immanuel.rl_agent.Q_table = bradley.opp_simple_agent.Q_table
#     # pikl_q_table(immanuel, immanuel_agent_q_table_path)
#     # print('finished training')


#     # immanuel = init_agent(small_chess_data, 'B') # training on sample of data
#     # bootstrap_agent(immanuel, immanuel_agent_q_table_path)
#     # immanuel.rl_agent_vs_simple_agent(immanuel_rl_agent_vs_simple_agent_training_results_filepath, training_length_1_000_games)
#     # pikl_q_table(immanuel, immanuel_agent_q_table_path)
#     # # print('finished training')


#     # play against agent
#     # play_game(bradley)

#     # agent v agent
#     # agent_vs_agent(bradley, immanuel)
