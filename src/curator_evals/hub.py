from typing import Dict, Any
from datasets import load_dataset, Dataset, DatasetDict

def upload_metrics(metrics: Dict[str, Any], model_config: Dict[str, Any], task: str = None, debug: bool = False):
    """
    Upload evaluation metrics to HuggingFace Hub.
    
    Args:
        metrics: Dictionary containing the evaluation metrics
        model_id: Identifier of the model being evaluated
        task: Name of the evaluation task
    """
        ## try to download the dataset from huggingface 
    try:
        data = load_dataset("collinear-ai/curator_evals_metrics", task, split="default")
    except:
        print(f"Dataset {task} not found on HuggingFace Hub, creating new dataset")
        data = None 

    ## if the dataest exists, add a new row to it
    new_model_name = model_config["model_id"] + "_" + model_config["model_type"] 
    if debug: 
        new_model_name = new_model_name + "_debug"
        
    new_row = {
        "model_name": new_model_name,
        **metrics
    }

    if data is not None:
        # Check if model_id already exists in dataset
        existing_idx = next((i for i, item in enumerate(data) if item["model_name"] == new_model_name), None)
        
        if existing_idx is not None:
            # Replace existing row
            data = data.select(list(range(existing_idx)) + list(range(existing_idx + 1, len(data))))
            data = data.add_item(new_row)
        else:
            # Add new row
            data = data.add_item(new_row)
    else:
        data = Dataset.from_list([new_row])
    
    data = DatasetDict({"default": data})
    data.push_to_hub("collinear-ai/curator_evals_metrics", task)


        