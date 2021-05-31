from flask import Flask, request, send_from_directory
from app.models import TransformerModel

model = TransformerModel()
app = Flask(__name__, static_url_path='')

@app.route('/recipe', methods=['POST'])
def create_instructions():
    ingredient_strings = request.json['ingredients']
    return model.get_instructions(ingredient_strings)

@app.route('/ingredients', methods=['GET'])
def get_ingredients():
    return {"ingredients" : model.get_ingredients()}


@app.route('/public/<path:path>', methods=['GET'])
def send_file(path):
    return send_from_directory('public', path)

@app.route('/', methods=['GET'])
def send_index():
    return send_from_directory('public', 'index.html')

