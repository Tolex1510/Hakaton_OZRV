# some imports
from flask import Flask, request, abort, make_response
from flask_cors import CORS, cross_origin
from Modules.parser import parse, parse_csv_arr, return_csv_response
from os import getenv
from dotenv import load_dotenv

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

    # with open(f"Uploads/{request.headers["filename"]}", 'w') as f:
    #     f.write(request.data.decode())

    with open(f"Uploads/{request.headers["filename"]}", 'r') as f:
        data = f.read()

    return data

# Get csv file without incidents, with Neuro create incidents and return csv's response with incidents
def parser(headers):
    """Get csv file without incidents, with Neuro create incidents and return csv's response with incidents"""
    csv_arr = parse(f"Uploads/{headers["filename"]}")

    csv_arr_with_incidents = func_from_ai(csv_arr)

    done_csv = parse_csv_arr(csv_arr)

    response = return_csv_response(done_csv)

    # response.headers["content-type"] = "text/csv"

    return response

def func_from_ai(data):
    pass

# start listening
def main():
    app.run(host="0.0.0.0", port=6080, debug=True)

if __name__ == "__main__":
    main()