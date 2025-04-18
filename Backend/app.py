# some imports
from flask import Flask, request, abort

app = Flask(__name__)

# start analys from AI
@app.route("startAnalys")
def startAnalys():
    if request.headers["X-auth"] != "bogdan_krasnov_luchshe_vseh_kak_parol":
        abort(404)

    pass # here we start func from AI and return "test_checks_example.csv" to Frontend

# start listening
def main():
    app.run(host="0.0.0.0", port=6080)

if __name__ == "__main__":
    main()