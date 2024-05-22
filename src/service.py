# service.py

from flask import Flask, Response, jsonify
import json


def run_prediction(model_type, tracks_per_list):
    if model_type == "advanced":
        ...
    elif model_type == "base":
        ...
    ...


def create_application() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return ...  # documentation

    @app.route("/predict/model/<model_type>/<int:tracks_per_list>", methods=["GET"])
    def predict_model(model_type, tracks_per_list):
        ...
        try:
            run_prediction(model_type, tracks_per_list)
            return jsonify({"model_type": model_type, "tracks_per_list": tracks_per_list})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route("/send_data/<data_type>", methods=["POST"])
    def send_data(data_type):
        ...
        return jsonify({"data_type": data_type})

    return app


if __name__ == "__main__":
    app = create_application()
    app.run(host="127.0.0.1", port=5000, debug=True)
