# some imports
from flask import Flask, request, abort, make_response, render_template
from flask_cors import CORS, cross_origin
from Modules.parser import parse
from os import getenv
from dotenv import load_dotenv
from Neuro.main import main as get_incidents
import pandas as pd
import csv

app = Flask(__name__)
CORS(app)

load_dotenv()

AUTH_PASS = getenv("AUTH_PASS")

# start analys from AI
@app.route("/startAnalys", methods=["POST"])
@cross_origin()
def startAnalys():
    try:
        if request.headers["X-auth"] != AUTH_PASS:
            abort(401)
    except:
        abort(401)

    request.files["file"].save(f"Uploads/{request.headers["filename"]}")

    # incidents = get_incidents(f"Uploads/{request.headers["filename"]}") # [[time, incident]]

    for_debug(f"Uploads/{request.headers["filename"]}")

    # convert(f"Uploads/{request.headers["filename"]}", incidents)

    with open(f"Uploads/{request.headers["filename"]}", 'r') as f:
        data = f.read()

    return data

@app.route("/")
@cross_origin()
def main_page():
    return render_template("index.html")

# convert arr like ['time', 'incident'] to csv file with same name in *path*
def convert(path, incidents):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(incidents)

def for_debug(path):
    data = pd.read_csv(path)[["time", "incident"]]

    data.to_csv(path)

# start listening
def main():
    app.run(host="0.0.0.0", port=6080, debug=True)

if __name__ == "__main__":
    main()