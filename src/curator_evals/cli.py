import argparse
import hashlib
import os
import json
from datetime import datetime
from typing import Dict
from datasets import load_dataset
from .models import ModelFactory
from .metrics import get_metrics
from .hub import upload_metrics
import asyncio
import pickle

async def evaluate(task: str, model_info: Dict, debug: bool = False):
    """
    Main evaluation function that orchestrates the evaluation process.
    
    Args:
        task: Name of the evaluation task (e.g., 'coherence_evals')
        model_info: Dictionary containing model configuration
    """

    # Load the dataset for the specific task
    data = load_dataset("collinear-ai/curator_evals_bench", task, split="default")
    if debug:
        print("Debug mode enabled, selecting 100 examples")
        data = data.shuffle(seed=42)
        data = data.select(range(100))
    
    print(f"Loaded dataset for task: {task} with {len(data)} examples")

    # Initialize the model using the factory
    model = ModelFactory.create_model(model_info)
    
    # Run evaluation
    results = await model.evaluate(data)

    # Calculate metrics
    metrics = get_metrics(results, task)
        
    # Upload metrics to hub
    # upload_metrics(metrics, model_info, task, debug)
    
    # Print metrics
    print(metrics)
    
    return results, metrics

async def main_async():
    parser = argparse.ArgumentParser(description="Run evaluations on the Curator Eval Bench")
    parser.add_argument("--task", required=True, help="Evaluation task to run")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--adapter-name", required=False, help="Adapter identifier on HuggingFace Hub")
    parser.add_argument("--model-type", required=True, choices=["llm"], #"llm_adapter", "classifier", "ensemble", "nvidia_complexity", "regex_math", "length_heuristic", "ppl_heuristic"], 
                       help="Type of model being evaluated")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--use-server", action="store_true", help="Use vLLM server mode")
    parser.add_argument("--server-url", default="http://localhost:8000", help="URL of the vLLM server")
    parser.add_argument("--input-format", default=None, help="Input format for the model")
    parser.add_argument("--output-format", default=None, help="Output format for the model (defaults to input-format if not specified)")
    parser.add_argument("--provider", default="vllm", help="Provider for the model")
    parser.add_argument("--api-key", default=None, help="API key for the model")
    args = parser.parse_args()
    
    model_info = {
        "model_id": args.model,
        "model_type": args.model_type,
        "adapter_name": args.adapter_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "use_server": args.use_server,
        "server_url": args.server_url,
        "task": args.task,
        "input_format": args.input_format,
        "output_format": args.output_format,
        "provider": args.provider,
        "api_key": args.api_key,
        "model": args.model
    }
    
    # await evaluate(args.task, model_info, args.debug)
    results, metrics = await evaluate(args.task, model_info, args.debug)
    
    # Save results to JSON file
    import json
    filename = f"results_{args.task}_{args.model.replace('/', '_').replace('-', '_')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")

def main():
    """Entry point for the CLI"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
