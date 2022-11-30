class Settings:
    """A class to store all settings for BradleyChess."""
    def __init__(self):
        self.learn_rate = 0.4
        # self.num_games = 10   # how many games to train the agent on, essentially how many times to run the main loop in the train_rl_agent method
        self.discount_factor = 0.7
        self.chance_for_greedy = 0.75
        self.chance_for_naive = 0.90
        self.num_columns = 75     # (75 per player)   
        self.new_move_pts = 1_000
        self.initial_q_val = 100
        # self.chessDB_sample_size = 100_000  # number of games to use during training and game
        # self.stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"