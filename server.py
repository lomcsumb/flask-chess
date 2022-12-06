from components import app
from flask_cors import CORS

if __name__ == "__main__":
    CORS(app.run(debug=True))