from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoProcessor, pipeline, BitsAndBytesConfig
from flask_restful import Resource, Api
from flask import Flask, request, jsonify

import torch
import logging
import gc

class LlavaTransformersPipeline(Resource):

    def __init__(self, config):
        self.model_name = config["model_name"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = None  
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load the model from library Transformers. Download if new in pc."""

        if self.pipe is None:
            try:
                logging.info(f"Loading model: {self.model_name} on {self.device}")
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
                model.to(self.device)

                self.processor = AutoProcessor.from_pretrained(self.model_name)

                self.pipe = pipeline(
                    "image-to-text",
                    model=model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    torch_dtype=self.torch_dtype,
                    load_in_8bit=True,
                    device=0 if torch.cuda.is_available() else -1,  # Ensure correct device usage
                )
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading model due to error {e}")
                raise RuntimeError("Model loading failed")

            return self.pipe


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

            inputs = self.processor(images=input_images, text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.pipe(
                    inputs
                )

            response = output[0]
        except Exception as e:
            logging.error(f"Error sending prompt {prompt} to model, due to error: {e}")
            return jsonify({"error": "Failed to process input"}), 500

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

