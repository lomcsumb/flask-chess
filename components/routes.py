###
import pandas as pd
import random
import time
import copy
from flask import render_template, request, redirect, jsonify
from components import app
from components.ml.Bradley import *
# from components.ml.src.helper_methods import init_agent
# from components.controller import * 
from components.chess_model import *

# chess_data = pd.read_pickle('components/kaggle_chess_data.pkl', compression = 'zip')

# q_table_path = r"D:\Users\Lom\Documents\Github\Capstone Project\White_Q_table_100_games.pkl"
# chess_data_path = r"D:\Users\Lom\Documents\Github\Capstone Project\kaggle_chess_data.pkl"

# q_table_path = r"bradley_agent_q_table.pkl"
# chess_data_path = r"small_chess_data.pkl"

# read pkl file that contains the dataframe.
# big_chess_DB = pd.read_pickle(chess_data_path, compression = 'zip')
big_chess_DB = pd.read_pickle('components/small_chess_data.pkl', compression = 'zip')
# chess_data = big_chess_DB.sample(100_000)
chess_data = big_chess_DB.sample(20000)

player = init_agent(chess_data)
player = Bradley(chess_data, 'W')

# load the agent with a previously trained agent's Q table
# don't train an agent when a user want to play the game, very time-consuming
#player.rl_agent.Q_table = pd.read_pickle(q_table_path, compression = 'zip') # pikl files load faster and the formatting is cleaner
player.rl_agent.Q_table = pd.read_pickle('components/bradley_agent_q_table.pkl', compression = 'zip') # pikl files load faster and the formatting is cleaner
player.rl_agent.is_trained = True # set this to trained since we assigned a preexisting Q table to new RL agent

# controller = StartGame(copy.deepcopy(player))
player_copy = copy.deepcopy(player)
# controller = StartGame(player_copy)

@app.route("/")
def index():
    return {"DC_Trinity": ["Superman", "Batman", "Wonderwoman"]}

@app.route("/startgame")
def startgame():
    # player = PlayerHands()


    # player = init_agent(chess_data)
    # player = Bradley(chess_data, 'W')

    # load the agent with a previously trained agent's Q table
    # don't train an agent when a user want to play the game, very time-consuming
    # player.rl_agent.Q_table = pd.read_pickle(q_table_path, compression = 'zip') # pikl files load faster and the formatting is cleaner
    # player.rl_agent.Q_table = pd.read_pickle('components/bradley_agent_q_table.pkl', compression = 'zip') # pikl files load faster and the formatting is cleaner
    # player.rl_agent.is_trained = True # set this to trained since we assigned a preexisting Q table to new RL agent

    # global controller
    # controller = StartGame(copy.deepcopy(player))
    # controller = PlayerHands(board)
    # controller.start_game(board)
    # legal_moves = controller.boardState.load_legal_moves_list()
    # legal_moves = load_legal_moves_list(controller.board)
    # controller.clearBoard()
    # player_copy = copy.deepcopy(player)
    # controller.setBoard(player_copy)

    global controller
    controller = StartGame(copy.deepcopy(player_copy))
    chess_move = controller.player.rl_agent_chess_move()
    chess_move_str = chess_move['chess_move_str']
    chess_move_src = chess_move['move_source']
    legal_moves = controller.player.get_legal_moves()
    fen_string = controller.player.get_fen_str()
    # player_turn = controller.play.playerTurnMessage()
    returnObj = {
            "legal_moves": legal_moves,
            # "player_turn": player_turn,
            "gameStarted": True,
            # "best_move": random.choice(legal_moves),
            "fen_string": fen_string,
            "best_move": chess_move_str,
            "ascii": str(controller.player.get_chessboard())
            }
    print(returnObj)
    return jsonify(returnObj)

@app.route("/endgame")
def endgame():
    return {"legal_moves": [],
            "player_turn": "Players has ended the game early",
            "gameStarted": False,
            "ascii": str(controller.player.get_chessboard())
            }

@app.route("/getmoves")
def getmoves():
    # legal_moves = load_legal_moves_list(controller.board)
    chess_move = controller.player.rl_agent_chess_move()
    chess_move_str = chess_move['chess_move_str']
    # chess_move_str = controller.agentmove['chess_move_str']
    # chess_move_src = chess_move['move_source']
    legal_moves = controller.player.get_legal_moves()
    fen_string = controller.player.get_fen_str()
    # player_turn = controller.play.playerTurnMessage()
    returnObj = {"legal_moves": legal_moves,
                # "player_turn": player_turn,
                "best_move": chess_move_str,
                "fen_string": fen_string,
                "ascii": str(controller.player.get_chessboard())
                }

    print(returnObj)
    return jsonify(returnObj)

@app.route("/playermoves")
def playermoves():
    fen_string = controller.player.get_fen_str()
    print(fen_string)
    # player_turn = controller.play.playerTurnMessage()
    return {
            "fen_string": fen_string,
            "ascii": str(controller.player.get_chessboard())
            }

# @app.route("/moveagent", methods=['GET', 'POST'])
# def moveagent():
#     if request.method == 'POST':
#         controller.agentmove = controller.player.rl_agent_chess_move()
#         return "Success"

# @app.route("/movewhite", methods=['POST'])
# def movewhite():
#     if request.method == 'POST':
#         move = request.json()
#         # controller.playerInput(move)
#         # controller.board.push_san(move)

#         print(move)
#         # print(controller.board)
#         return "Success"

@app.route("/moveblack", methods=['POST'])
def moveblack():
    if request.method == 'POST':
        move = request.json
        # controller.playerInput(move)
        controller.player.recv_opp_move(str(move))
        print(move)
        # print(controller.board)
        returnObj = {'Success':True}
        return jsonify(returnObj)
    # else:
    #     return jsonify({'Success':False})
# @app.route("/movewhite", methods=['GET','POST'])
# def movewhite():
#     if request.method == 'POST':
#         move = request.json
#         # controller.playerInput(move)
#         controller.board.push_san(move)
#         print(move)
#         print(controller.board)
#         return "Success"

@app.route("/board")
def board():
    return {"DC_Trinity": ["Superman", "Batman", "Wonderwoman"]}



# import pandas as pd
# from flask import render_template, request, redirect
# from dash_package.dashboard import app
# from dash_package.functions import *

# from dash_package.database import conn

# @app.server.route('/dash')
# def dashboard():
#         return app.index()

# @app.server.route('/')
# def index():
#         #query = 'SELECT * FROM test_data LIMIT 5;'
#         df = pd.DataFrame()
#         #df = pd.read_sql(query, conn)
#         return render_template('index.html', dataSaved=False, dataFound=False, data=df, next_w='e4', next_b='')

# @app.server.route('/test', methods=['GET', 'POST'])
# def test():
#         if request.method == 'POST':
#                 move = request.form['test']
#                 query = 'SELECT * FROM test_data WHERE B1= \'' + move + '\' LIMIT 1;'
#                 df = pd.read_sql(query, conn)
#                 return render_template('index.html', dataSaved=False, dataFound=False, data=df, next_w=df.iloc[0]['W2'], next_b='')

# @app.server.route('/returning', methods=['GET', 'POST'])
# def returning():
#         if request.method == 'POST':
#                 move = request.form['returning']
#                 query = 'SELECT * FROM test_data WHERE W1= \'' + move + '\' LIMIT 1;'
#                 df = pd.read_sql(query, conn)
#                 return render_template('index.html', dataSaved=False, dataFound=False, data=df, next_w='', next_b=df.iloc[0]['B1'])

# @app.server.route('/save', methods=['GET', 'POST'])
# def save():
#         if request.method == 'POST':
#                 game = request.form['save']
#                 data = [game]
#                 df = pd.DataFrame(data, columns=['Games'])
#                 df.to_sql('test_data', conn,if_exists='append',index=False)
#                 return render_template('index.html', dataSaved=True, dataFound=False, data=df, next_w='e4', next_b=df['B1'])

# @app.server.route('/search', methods=['GET', 'POST'])
# def search():
#         if request.method == 'POST':
#                 game = request.form['search']
#                 print(game)
#                 query = 'SELECT * FROM test_data WHERE Games = \'' + game + '\'' 
#                 #query = 'SELECT * FROM test_data LIMIT 5;'
#                 print(query)
#                 df = pd.read_sql(query, conn)
#                 return render_template('index.html', dataSaved=False, dataFound=True, data=df, next_w='e4', next_b='')