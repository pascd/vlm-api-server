import torch
import logging
import sys
import traceback

# Configure verbose logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('deepseek_diagnostic.log')
                    ])

def diagnose_deepseek():
    try:
        # Import DeepSeek libraries
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

        # Print Python and library versions
        print(f"Python Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        # Try to load the model
        model_name = "deepseek-ai/deepseek-vl-7b-base"
        
        print("\n--- Loading Processor ---")
        processor = VLChatProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        print("Processor loaded successfully")
        
        print("\n--- Loading Model ---")
        model = MultiModalityCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        print("Model loaded successfully")
        
        # Print model details
        print("\n--- Model Details ---")
        print(f"Model Type: {type(model)}")
        print(f"Device: {model.device}")
        
        # Try to get model generation method
        print("\n--- Checking Generation Method ---")
        if hasattr(model, 'generate'):
            print("Generate method exists")
        else:
            print("No generate method found")
        
        # Attempt to introspect the model
        print("\n--- Model Introspection ---")
        print(model)
        
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_deepseek()