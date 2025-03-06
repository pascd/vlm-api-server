from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoProcessor, pipeline, BitsAndBytesConfig
from flask_restful import Resource, Api
from flask import Flask, request, jsonify

import torch
import logging
import gc
from PIL import Image

class LlavaTransformersPipeline(Resource):

    def __init__(self, config):
        self.model_name = config["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = None  
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load the model and distribute across multiple GPUs efficiently."""
        if self.pipe is None:
            try:
                logging.info(f"Loading model: {self.model_name} on {self.device}")
                
                # Enable memory-efficient loading with 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype
                )
                
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    quantization_config=bnb_config
                )
                
                # If multiple GPUs are available, use DataParallel
                if torch.cuda.device_count() > 1:
                    logging.info(f"Using {torch.cuda.device_count()} GPUs!")
                    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
                    model.to(f"cuda:{model.device_ids[0]}")  # Move model to first GPU
                else:
                    model.to(self.device)
                
                self.processor = AutoProcessor.from_pretrained(self.model_name)

                self.pipe = pipeline(
                    "image-to-text",
                    model=model.module if isinstance(model, torch.nn.DataParallel) else model,
                    tokenizer=self.processor.tokenizer,
                    image_processor=self.processor.image_processor,
                    torch_dtype=self.torch_dtype,
                    #device=0 if torch.cuda.is_available() else -1,
                )
                
                logging.info("Model loaded successfully with multi-GPU support.")
            except Exception as e:
                logging.error(f"Error loading model due to error {e}")
                raise RuntimeError("Model loading failed")

    def create_model_prompt(self, prompt_text):
        """Formats the prompt in a chat-like structure"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def send_prompt(self, input_text, input_images=None):
        """ Send the prompt to the model """
        try:
            conversation = self.create_model_prompt(prompt_text=input_text)
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Ensure images are in correct format
            processed_images = []
            if input_images:
                for img in input_images:
                    if isinstance(img, Image.Image):  # If already PIL image
                        processed_images.append(img)
                    elif isinstance(img, str) and img.strip():  # Base64 string
                        try:
                            image_data = base64.b64decode(img)
                            image = Image.open(BytesIO(image_data)).convert("RGB")
                            processed_images.append(image)
                        except Exception as e:
                            logging.error(f"Error decoding Base64 image: {e}")
                            return {"error": "Invalid image format"}, 400
                    else:
                        logging.error(f"Unsupported image format: {type(img)}")
                        return {"error": "Unsupported image format"}, 400
            
            # Process inputs for model
            if not processed_images:
                # Create a blank white image as a placeholder
                blank_image = Image.new('RGB', (224, 224), (255, 255, 255))
                processed_images.append(blank_image)
            
            if processed_images:
                inputs = self.processor(images=processed_images, text=prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.pipe(inputs)
            
            response = output[0] if isinstance(output, list) and len(output) > 0 else output
            return response, 200
        
        except Exception as e:
            logging.error(f"Error sending prompt '{input_text}' due to error: {e}")
            return {"error": "Failed to process input"}, 500
        
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def get(self):
        """ Returned the loaded model """
        return jsonify({"model_name": self.model_name}), 200
    
    def post(self):
        """ Handles POST request for processing text and images """
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "Missing required parameter: text"}), 400
            
            input_text = data['text']
            input_images = data.get('images', None)

            return self.send_prompt(input_text, input_images)

        except Exception as e:
            logging.error(f"Error in POST request: {str(e)}")
            return jsonify({"error": "Server error"}), 500
