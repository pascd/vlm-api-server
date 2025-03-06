import torch
import logging
import gc
import os
import sys
import traceback
from transformers import AutoTokenizer, AutoConfig, PreTrainedModel

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeekR1TransformersPipeline:
    def __init__(self, config):
        # Force disable any FP8 stuff right at the beginning
        os.environ["TRANSFORMERS_NO_FP8"] = "1"
        os.environ["ACCELERATE_DISABLE_FP8"] = "1"
        os.environ["USE_FLASH_ATTENTION"] = "0"
        
        # This is critical - disable the FP8 module entirely
        sys.modules["transformers.quantizers.quantizer_finegrained_fp8"] = None
        
        self.model_name = config["model_name"]
        self.model = None
        self.tokenizer = None
        
        # Check GPU availability for logging purposes
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)[0]
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            logging.info(f"Using CUDA device: {device}")
            logging.info(f"GPU Compute Capability: {capability}")
            logging.info(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
            logging.info(f"Free GPU Memory: {free_memory / (1024**3):.2f} GB")
            
            self.device = "cuda:0"
        else:
            self.device = "cpu"
            logging.info("No GPU available, using CPU")
        
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        if self.model is None:
            self.load_model()

    def load_model(self):
        try:
            logging.info(f"🚀 Loading model: {self.model_name}")
            torch.cuda.empty_cache()
            gc.collect()
            
            # First, load the tokenizer which should be safe
            logging.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Use a totally different approach - load from a different model architecture
            # that doesn't have FP8 quantization
            logging.info("Attempting to use alternative model loading approach...")
            
            try:
                # Try importing and using DirectLLMEngine
                from optimum.llm import LLMConfig, DirectLLMEngine
                
                # Create a configuration for direct loading
                config = LLMConfig.from_pretrained(self.model_name)
                
                # Load the model directly without going through HF's pipeline
                self.model = DirectLLMEngine.from_pretrained(self.model_name, config=config)
                
                logging.info(f"✅ Model loaded successfully using DirectLLMEngine")
            except (ImportError, Exception) as e:
                logging.warning(f"DirectLLMEngine loading failed: {e}")
                
                # Fall back to CPU-only loading as a last resort
                logging.info("Falling back to CPU-only mode as a last resort")
                
                # Temporarily hide CUDA from PyTorch
                original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                
                try:
                    # Import AutoModelForCausalLM only after hiding CUDA
                    from transformers import AutoModelForCausalLM
                    
                    # Load the model to CPU first
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    # Restore CUDA visibility
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
                    
                    # Now manually move to GPU if available
                    if torch.cuda.is_available():
                        # Determine appropriate precision
                        capability = torch.cuda.get_device_capability(0)[0]
                        if capability >= 8:
                            target_dtype = torch.bfloat16
                        elif capability >= 7:
                            target_dtype = torch.float16
                        else:
                            target_dtype = torch.float32
                        
                        logging.info(f"Moving model from CPU to GPU with {target_dtype}...")
                        
                        # Move model to GPU with appropriate precision
                        self.model = self.model.to(device="cuda", dtype=target_dtype)
                        
                    logging.info(f"✅ Model loaded successfully with CPU loading + GPU transfer")
                except Exception as e:
                    logging.error(f"CPU loading also failed: {e}")
                    raise
            
            # Set model to evaluation mode
            if isinstance(self.model, PreTrainedModel):
                self.model.eval()
            torch.set_grad_enabled(False)

        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            logging.error(traceback.format_exc())
            
            logging.error("All loading attempts failed. Please consider using a different model or hardware.")
            raise RuntimeError(f"Model loading failed completely: {str(e)}")

    def send_prompt(self, input_text):
        try:
            logging.debug(f"Processing prompt: {input_text}")
            
            # DirectLLMEngine has a different interface compared to HF models
            if hasattr(self.model, 'generate_with_streaming'):
                # This is the DirectLLMEngine case
                outputs = []
                for output in self.model.generate_with_streaming(
                    input_text,
                    max_new_tokens=512,
                    temperature=0.0  # Equivalent to do_sample=False
                ):
                    outputs.append(output)
                
                # Join all outputs
                response_text = "".join(outputs)
            else:
                # Standard HF model case
                # Get device of model
                device = next(self.model.parameters()).device
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask', None),
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True
                    )
                
                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logging.info(f"Generated response: {response_text}")
            return {"response": response_text}

        except Exception as e:
            logging.error(f"❌ Error processing input: {e}")
            logging.error(traceback.format_exc())
            return {"error": str(e)}, 500

    def diagnose(self):
        try:
            info = {
                "model_name": self.model_name,
                "model_type": type(self.model).__name__ if self.model else "None",
                "is_cuda_available": torch.cuda.is_available(),
                "is_optimum_direct": hasattr(self.model, 'generate_with_streaming')
            }
            
            # Add details about model location if it's a standard HF model
            if not info["is_optimum_direct"] and self.model is not None:
                first_param = next(self.model.parameters(), None)
                if first_param is not None:
                    info["device"] = str(first_param.device)
                    info["dtype"] = str(first_param.dtype)
            
            return info
        except Exception as e:
            logging.error(f"Diagnosis failed: {e}")
            return {"error": str(e)}