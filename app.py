from flask import Flask, request, Response
from flask_cors import CORS
import jsonpickle
from inferencia import deteccion

app = Flask(__name__)
CORS(app)

@app.route('/detection', methods=['POST'])
def detection():
    r = request

    kangaroo_detector = deteccion("models/model2.pth")
    
    # predecimos
    result = kangaroo_detector.predice(r)

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1005)
