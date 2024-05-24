#!/usr/bin/env python3
# service.py

from flask import Flask, Response, jsonify
from src.models_functions import train_model


def run_prediction(model_type: str, tracks_per_list: int):
    if model_type == "advanced":
        result = train_model.train_advanced_model(tracks_per_list)
        return result
    elif model_type == "base":
        ...
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def create_application() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return "Hello World!"

    @app.route("/predict/model/<model_type>/<int:tracks_per_list>", methods=["GET"])
    def predict_model(model_type: str, tracks_per_list: int):
        try:
            result = run_prediction(model_type, tracks_per_list)
            result = result.to_dict()
            return jsonify({"model_type": model_type, "tracks_per_list": tracks_per_list, "result": result})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route("/send_data/<data_type>", methods=["POST"])
    def send_data(data_type: str):
        return jsonify({"data_type": data_type})

    return app


if __name__ == "__main__":
    app = create_application()
    app.run(host="127.0.0.1", port=5000, debug=True)
