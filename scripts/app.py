from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import logging
import os
import json
import torch
import base64
from io import BytesIO
from PIL import Image
from llava_transformers_pipe import LlavaTransformersPipeline
from deepseek7b_transformers_pipe import DeepSeek7bTransformersPipeline
from deepseekr1_transformers_pipe import DeepSeekR1TransformersPipeline

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
except Exception as e:
    logging.error(f"Failed to load config file: {e}")
    config = {"model_name": "deepseek-ai/deepseek-vl-7b-base", "port": 5000}

# Initialize Flask app
app = Flask(__name__)
api = Api(app)
CORS(app)  # Enable CORS if needed

# Class just to test if the API is available
class Test(Resource):
    def get(self):
        return {"message": "API is working."}, 200

# Test resource of the API at the root endpoint
api.add_resource(Test, '/')

# Initialize the model
match config["model_name"]:
    case "llava-hf/llava-v1.6-34b-hf":
        model = LlavaTransformersPipeline(config)
    case "deepseek-ai/deepseek-vl-7b-base":
        model = DeepSeek7bTransformersPipeline(config)
    case "deepseek-ai/deepseek-r1":
        model = DeepSeekR1TransformersPipeline(config)

def process_images(image_list):
    """Helper function to process images (base64 decoding or URLs)."""
    processed_images = []
    if image_list:
        for img in image_list:
            if isinstance(img, str) and img.strip():  # Base64 string
                try:
                    if img.startswith("data:image"):  # Remove metadata header if present
                        img = img.split(",")[1]
                    image_data = base64.b64decode(img)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    processed_images.append(image)
                except Exception as e:
                    logging.error(f"Error decoding Base64 image: {e}")
                    return {"error": "Invalid image format"}, 400
            elif isinstance(img, Image.Image):  # If already a PIL image
                processed_images.append(img)
            else:
                logging.error(f"Unsupported image format: {type(img)}")
                return {"error": "Unsupported image format"}, 400
    return processed_images

@app.route("/chat", methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400
        
        input_text = data["text"]
        input_images = data.get("images", [])  # Get the list of images (default to empty list)
        
        # Process the images only if they are provided
        processed_images = None
        if input_images:  # If images are provided
            processed_images = process_images(input_images)
            if isinstance(processed_images, dict):  # Error in image processing
                return jsonify(processed_images), 400
        
        # Send the text and (optionally) processed images to the model
        response, status_code = model.send_prompt(input_text, processed_images)
        return jsonify(response), status_code
    
    except Exception as e:
        logging.error(f"❌ Error in POST request: {e}")
        return jsonify({"error": "Server error"}), 500

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
