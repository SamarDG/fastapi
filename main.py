from flask import Flask
from flask import request

app = Flask(__name__)

def ML_code(a):
    return a

@app.route('/')
def home():
    return '200 OK!! Connected to Data Source Config Page'


#http://127.0.0.1:5000/query/?url=https://google.com
@app.route('/query/')
def func1():
    query = request.args.get('url')
    return ML_code(query)

if __name__ == '__main__':
    app.run(debug=True)
