from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from PIL import Image
from deepseek7b_transformers_pipe import DeepSeek7bTransformersPipeline
import torch
import numpy as np
import logging
import os
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    config = {
        "model_name": "deepseek-ai/deepseek-vl-7b-base",
        "port": 5000
    }

# Initialize Flask app
app = Flask(__name__)
api = Api(app)
CORS(app)  # Enable CORS if needed

# Class just to test if the API is available
class Test(Resource):
    def get(self):
        return {"message": "DeepSeek VL API is working."}, 200

# Test resource of the API at the root endpoint
api.add_resource(Test, '/')

# Initialize the model
model = DeepSeek7bTransformersPipeline(config)

@app.route("/chat", methods=['GET'])
def get_model():
    """Returns the loaded model information."""
    return {"message": f"Model loaded {config['model_name']}"}, 200

@app.route("/chat", methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400
        
        response = model.send_prompt(data["text"], data.get("images"))
        return jsonify(response) if isinstance(response, dict) else jsonify(response[0]), response[1]
    
    except Exception as e:
        logging.error(f"❌ Error in POST request: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route("/diagnose", methods=['GET'])
def diagnose():
    """Provide diagnostic information about the model."""
    try:
        diag_info = model.diagnose()
        return jsonify(diag_info), 200
    except Exception as e:
        logging.error(f"Diagnosis failed: {e}")
        return jsonify({"error": str(e)}), 500

# Diagnostics for PyTorch and CUDA
def print_cuda_info():
    logging.info("CUDA Information:")
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logging.info(f"Current CUDA Device: {torch.cuda.current_device()}")
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# Print CUDA info on startup
print_cuda_info()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.get("port", 5000), debug=True, use_reloader=False)