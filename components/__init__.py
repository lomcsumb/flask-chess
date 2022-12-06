from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)  # Latest addition

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['DEBUG'] = True



# app = Flask(__name__)
# app.config['suppress_callback_exceptions']= True

from components import routes