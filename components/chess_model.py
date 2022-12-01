import chess
from components.ml.Bradley import *

# class BoardState:
#     def __init__(self):
#         self.board = chess.Board()
#         # self.isPlayer1 = True
#         # self.legalMoves = []

#     # def switch_player(self):
#     #     self.isPlayer1 = not self.isPlayer1

#     ##
#     def load_legal_moves_list(self):
#         legal_moves = []
#         for i in self.board.legal_moves:
#             legal_moves.append(i)
        
#         i = 0
#         num_legal_moves = self.board.legal_moves.count()
#         legal_moves_str = []
#         while num_legal_moves > 0:
#             legal_moves_str.append(self.board.san(legal_moves[i]))
#             i += 1
#             num_legal_moves -= 1
            
#         return legal_moves_str

#     ###
#     def print_legal_moves(self ):
#         print('something')

class PlayerHands:
    def __init__(self):
        self.isPlayer1 = True
        self.currentMove = ''

    def switch_player(self):
        self.isPlayer1 = not self.isPlayer1

    def find_player_turn(self):
        if(self.isPlayer1):
            return "White"
        else:
            return "Black"
    
    def playerTurnMessage(self):
        return f'It is {self.find_player_turn()}\'s turn to move. Below are avaialable moves for white:\n'

    def availableMoveMessage(self):
        return 'something'

    def sendLegalMoves(self, board):
        return board

    def setPlayerInput(self, playerInput):
        self.currentMove = playerInput

    # def start_game(self, boardState):
    #         # W_turn = True 
    #         while not boardState.board.is_game_over():
    #             # print(f'It is {player_turn}\'s turn to move\n')
    #             # print(f'available moves for {player_turn} are:\n')
    #             # print(self.load_legal_moves_list(board))
    #             # print(f'available moves for ')
    #             # print(boardState.load_legal_moves_list)
    #             # chess_move = str(input('\nenter chess mov str, or enter q to quit'))
    #             # chess_move = self.playerInput
    #             if self.currentMove == 'q': break
                
    #             try:
    #                 boardState.board.push_san(self.currentMove)
    #             except ValueError:
    #                 print('wrong move, try again')
    #                 continue

    #             boardState.switch_player()

    #             # print(f'{player_turn} played {chess_move}')


def load_legal_moves_list(board):
    legal_moves = []
    for i in board.legal_moves:
        legal_moves.append(i)
        
    i = 0
    num_legal_moves = board.legal_moves.count()
    legal_moves_str = []
    while num_legal_moves > 0:
        legal_moves_str.append(board.san(legal_moves[i]))
        i += 1
        num_legal_moves -= 1
            
    return legal_moves_str

# class StartGame:
#     def __init__(self, playerHands):
#         self.stop = False
#         self.play = playerHands
#         self.board = chess.Board()

#     def endGame(self):
#         self.stop = not self.stop

    
#     def checkStatus(self, playerHands, boardState):
#         chess_move = playerHands.currentMove
#         if chess_move == 'q':
#             self.endGame
        # while not boardState.board.is_game_over():
        #     if self.currentMove == 'q': break

        #     try:
        #         boardState.board.push_san(self.currentMove)
        #     except ValueError:
        #         print('wrong move, try again')
        #         continue
            
        #     boardState.switch_player()

class StartGame:
    def __init__(self, player):
        self.stop = False
        self.player = player
        self.agentmove = pd.DataFrame()
        # self.board = chess.Board()

    def endGame(self):
        self.stop = not self.stop

    
    def checkStatus(self, player, boardState):
        chess_move = player.currentMove
        if chess_move == 'q':
            self.endGame