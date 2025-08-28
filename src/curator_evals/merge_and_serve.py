import argparse
import os
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import subprocess

def merge_and_serve(base_model: str, 
                    adapter_name: str, 
                    port: int = 8000, 
                    tensor_parallel_size: int = 1, 
                    pipeline_parallel_size: int = 1,
                    cache_dir: str = "/mnt/training",
                    force_redownload: bool = False):
    """
    Merge a base model with an adapter and start a vLLM server with the merged model.
    
    Args:
        base_model: Base model identifier on HuggingFace Hub
        adapter_name: Adapter identifier on HuggingFace Hub
        port: Port to run the vLLM server on
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
    """
    # Create a unique hash for this model+adapter combination
    model_hash = hashlib.md5(f"{base_model}_{adapter_name}".encode()).hexdigest()
    merged_model_dir = os.path.join(cache_dir, "merged_models", model_hash)

    if not os.path.exists(merged_model_dir) or force_redownload:
        os.makedirs(merged_model_dir, exist_ok=True)
        print(f"Loading base model and merging adapter...")
        base_model_obj = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base_model_obj, adapter_name)
        
        # Merge adapter weights with base model
        print("Merging adapter weights with base model...")
        model = model.merge_and_unload()
        
        # Save the merged model
        print(f"Saving merged model to {merged_model_dir}...")
        model.save_pretrained(merged_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(merged_model_dir)
    else:
        print(f"Using cached merged model from {merged_model_dir}")

    # Start the vLLM server
    print(f"Starting vLLM server with merged model on port {port}...")
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--port", str(port),
        "--model", merged_model_dir,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--pipeline-parallel-size", str(pipeline_parallel_size),
    ]
    
    # Run the server in the foreground
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Merge a base model with an adapter and start a vLLM server")
    parser.add_argument("--base-model", required=True, help="Base model identifier on HuggingFace Hub")
    parser.add_argument("--adapter-name", required=True, help="Adapter identifier on HuggingFace Hub")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the vLLM server on")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Number of GPUs to use for pipeline parallelism")
    parser.add_argument("--cache-dir", type=str, default="/mnt/training", help="Directory to cache the merged model")
    parser.add_argument("--force-redownload", action="store_true", help="Force redownload of the merged model")
    args = parser.parse_args()
    
    merge_and_serve(args.base_model, args.adapter_name, args.port, args.tensor_parallel_size, args.pipeline_parallel_size, args.cache_dir, args.force_redownload)

if __name__ == "__main__":
    main() 