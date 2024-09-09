# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS

from Routing import getPath

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/')
def home():
    return jsonify({'message': 'Hello World'})
@app.route('/getRoute', methods=['POST'])
def index():
    data = request.get_json()
    print(data)
    start_w=(data['start_lat'], data['start_long'])
    end_w=(data['end_lat'], data['end_long'])
    path,distance= getPath(start_w,end_w)
    return jsonify({"Path": path,"Distance":distance})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)