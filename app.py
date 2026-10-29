from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Sruthi's first Azure Web App!"
