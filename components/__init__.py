from flask import Flask

server = Flask(__name__)
server.config['DEBUG'] = True


app = Flask(__name__)
# app.config['suppress_callback_exceptions']= True

from components import routes