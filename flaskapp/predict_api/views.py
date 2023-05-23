"""Python script for flask application"""
import json

from flask import Flask, jsonify, render_template, request
from predict_api.models import Content

app = Flask(__name__)


# a simple page page that says hello
@app.route("/hello")
def hello():
    return "Hello, World!"


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        content_value = {
            # Tourney info
            "tourney_info": {
                "surface": str(request.form.get("surface")),
                "match_num": int(request.form.get("match_num")),
                "tourney_date": str(request.form.get("tourney_date")),
                "best_of": int(request.form.get("best_of")),
                "tourney_level": request.form.get("tourney_level"),
            },
            # Player 1 info
            "player1": {
                "id": int(request.form.get("player1_id")),
                "age": float(request.form.get("player1_age")),
                "hand": str(request.form.get("player1_hand")),
                "ht": float(request.form.get("player1_ht")),
                "rank": float(request.form.get("player1_rank")),
                "rank_points": float(request.form.get("player1_rank_points")),
            },
            # Player 2 info
            "player2": {
                "id": int(request.form.get("player2_id")),
                "age": float(request.form.get("player2_age")),
                "hand": str(request.form.get("player2_hand")),
                "ht": float(request.form.get("player2_ht")),
                "rank": int(request.form.get("player2_rank")),
                "rank_points": float(request.form.get("player2_rank_points")),
            },
        }
        # apply the prediction function imported from Content class
        result = Content.predict(content_value)
        return render_template("predict_result.html", result=result)
    return render_template("predict_input.html")


@app.route("/predict/", methods=["POST"])
def predict_input():
    """Returns json HTML POST request. Input is json dict"""

    # request input
    input_json = request.data.decode()
    print(input_json)

    # convert the json input to a python dict
    content_value = json.loads(input_json)

    # apply the prediction function imported from Content class
    result = Content.predict(content_value)

    # return the result in json format
    return jsonify({"result": result})
