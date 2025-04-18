# some imports
from flask import Flask, request, abort, Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# start analys from AI
@app.route("/startAnalys", methods=["POST"])
@cross_origin()
def startAnalys():
    try:
        if request.headers["X-auth"] != "bogdan_krasnov_luchshe_vseh_kak_parol":
            abort(401)
    except:
        abort(401)

    with open("templates/test_checks_example.csv", mode='r') as f:
        file = f.read()

    return "hello"

# start listening
def main():
    app.run(host="0.0.0.0", port=6080, debug=True)

if __name__ == "__main__":
    main()