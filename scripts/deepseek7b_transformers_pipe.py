import torch
import logging
import gc
import traceback
from flask_restful import Resource

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# Configure more verbose logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeek7bTransformersPipeline:
    def __init__(self, config):
        # Explicitly set CUDA device
        self.device = self._get_best_device()
        
        self.model_name = config["model_name"]
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Explicitly load model to specific device
        self._ensure_model_loaded()
    
    def _get_best_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            # Find the GPU with the most free memory
            device = torch.cuda.current_device()
            
            # Force garbage collection and empty cache
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log GPU memory details
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            logging.info(f"Using CUDA device: {device}")
            logging.info(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
            logging.info(f"Free GPU Memory: {free_memory / (1024**3):.2f} GB")
            
            return f"cuda:{device}"
        return "cpu"

    def _ensure_model_loaded(self):
        """Ensures model is loaded only once and on the correct device."""
        if self.model is None:
            self.load_model()

    def load_model(self):
        """Load the DeepSeek-VL model and processor with explicit device management."""
        try:
            logging.info(f"🚀 Loading model: {self.model_name} on {self.device}")

            # Aggressive memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

            # Load processor 
            logging.debug("Loading processor...")
            self.processor = VLChatProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # Set tokenizer
            logging.debug("Loading tokenizer...")
            self.tokenizer = self.processor.tokenizer

            # Explicit dtype and device handling
            torch_dtype = torch.float32  # Use float32 to avoid dtype issues
            
            # Load model with explicit device mapping
            logging.debug("Loading model...")
            self.model = MultiModalityCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )

            # Move model to specific device
            #self.model.to(self.device)

            # Additional model optimization
            self.model.eval()  # Set to evaluation mode
            torch.set_grad_enabled(False)  # Disable gradient computation
            
            logging.info(f"✅ Model loaded successfully")

        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def create_model_prompt(self, prompt_text, input_images=None):
        """Create a conversation-style prompt for the model."""
        conversation = [
            {
                "role": "User",
                "content": prompt_text,
                "images": input_images if input_images else []
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        return conversation

    def send_prompt(self, input_text, input_images=None):
        """Process input text and images through the DeepSeek-VL model."""
        try:
            logging.debug(f"Processing prompt: {input_text}")
            logging.debug(f"Input images: {input_images}")

            # Create conversation
            conversation = self.create_model_prompt(prompt_text=input_text, input_images=input_images)
            
            # Process images if any
            pil_images = load_pil_images(conversation)
            logging.debug(f"Processed images: {pil_images}")

            # Prepare inputs using the processor's specific method
            logging.debug("Preparing model inputs...")
            inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                return_tensors="pt"
            )

            # Ensure all inputs are on the correct device with correct dtypes
            processed_inputs = {}
            for k, v in inputs.__dict__.items():
                if isinstance(v, torch.Tensor):
                    # Ensure input_ids is long dtype
                    if k == 'input_ids':
                        processed_inputs[k] = v.to(device=self.device, dtype=torch.long)
                    # Ensure attention_mask is bool
                    elif k == 'attention_mask':
                        processed_inputs[k] = v.to(device=self.device, dtype=torch.bool)
                    # Other tensors use model's default dtype
                    else:
                        processed_inputs[k] = v.to(device=self.device, dtype=self.model.dtype)
                else:
                    processed_inputs[k] = v

            # Debug input preparation
            logging.debug("Input preparation details:")
            for k, v in processed_inputs.items():
                if isinstance(v, torch.Tensor):
                    logging.debug(f"{k}: {v.shape}, {v.dtype}, {v.device}")

            # Generate response 
            logging.debug("Generating response...")
            with torch.no_grad():
                # Generate using standard generation method
                outputs = self.model.language_model.generate(
                    input_ids=processed_inputs['input_ids'],
                    attention_mask=processed_inputs.get('attention_mask', None),
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )

            # Decode response
            logging.debug("Decoding response...")
            response_text = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            logging.info(f"Generated response: {response_text}")
            return {"response": response_text}

        except Exception as e:
            logging.error(f"❌ Error processing input: {e}")
            logging.error(traceback.format_exc())
            return {"error": str(e)}, 500

    # Diagnostic method to check model status
    def diagnose(self):
        """Provide detailed diagnostic information about the model."""
        try:
            diag_info = {
                "model_name": self.model_name,
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "is_cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "current_cuda_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "model_type": type(self.model).__name__
            }
            return diag_info
        except Exception as e:
            logging.error(f"Diagnosis failed: {e}")
            return {"error": str(e)}