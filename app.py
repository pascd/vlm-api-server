from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from PIL import Image
from llava_transformers_pipe import LlavaTransformersPipeline

import torch
import numpy as np
import logging
import os
import json


# Get the project root for config loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
logging.info(f"Base DIR: {BASE_DIR}")

# Load config file
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    logging.info("Config file loaded successfully.")

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format: Expected a dictionary, got {type(config)}")
    
except Exception as e:
    logging.error(f"Failed to load config file: {e}")
    config = {}

# Initialize Flask app
app = Flask(__name__)
api = Api(app)
CORS(app)  # Enable CORS if needed

# Class just to test if the API is available
class Test(Resource):
    def get(self):
        return {"message": "Llava API is working."}, 200

# Test resource of the API at the root endpoint
api.add_resource(Test, '/')

# Register the Llava model as a resource with config
api.add_resource(LlavaTransformersPipeline, '/chat', resource_class_kwargs={"config": config})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config["port"], debug=True)
