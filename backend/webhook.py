# some imports
from flask import Flask, request

app = Flask(__name__)

# start analys from AI
@app.route("startAnalys")
def startAnalys():
    pass # here we start func from AI to start analys

# start listening
def main():
    app.run(host="0.0.0.0", port=6080)

if __name__ == "__main__":
    main()