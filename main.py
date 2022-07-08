from flask import Flask, request, Response
from flask_cors import CORS
import jsonpickle
from inferencia import deteccion
import requests
import os

app = Flask(__name__)
_here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(_here, 'models/model.pth')
CORS(app)

@app.route('/detection', methods=['POST'])
def detection():
    r = request

    kangaroo_detector = deteccion(filename)
    
    # predecimos
    result = kangaroo_detector.predice(r)

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")


def download_model():
    print(_here)
    filename = os.path.join(_here, 'models/model.pth')

    if os.path.isfile(filename):
        return
    else:
        model_url = "https://firebasestorage.googleapis.com/v0/b/master-ia-2279b.appspot.com/o/model.pth?alt=media"
        response = requests.get(model_url)
        open(filename, "wb").write(response.content)

if __name__ == '__main__':
    download_model()
    app.run(host='0.0.0.0', port=1005)
