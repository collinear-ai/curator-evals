from datasets import load_dataset, DatasetDict
def pairwise_to_binary(dataset):
    """
    Converts a dataset with pairwise comparisons (chosen/rejected responses) into a binary classification dataset.
    
    Args:
        dataset: A Dataset object containing 'id', 'profile_id', 'task', 'prompt', 'chosen' and 'rejected' fields
        
    Returns:
        Dataset: A new Dataset object with fields 'id', 'profile_id', 'task', 'prompt', 'response' and 'label',
                where each original example is split into two examples:
                - One with the chosen response (label=1)
                - One with the rejected response (label=0)
    """
    # Create new lists to store the flattened data
    new_data = {
        'id': [],
        'profile_id': [],
        'task': [],
        'prompt': [],
        'response': [],
        'label': []
    }
    
    # Iterate through each example in the dataset
    for example in dataset:
        # Add chosen response (label 1)
        new_data['id'].append(example['id'])
        new_data['profile_id'].append(example['profile_id'])
        new_data['task'].append(example['task'])
        new_data['prompt'].append(example['prompt'])
        new_data['response'].append(example['chosen'])
        new_data['label'].append(1)
        
        # Add rejected response (label 0)
        new_data['id'].append(example['id'])
        new_data['profile_id'].append(example['profile_id'])
        new_data['task'].append(example['task'])
        new_data['prompt'].append(example['prompt'])
        new_data['response'].append(example['rejected'])
        new_data['label'].append(0)
    
    # Create new Dataset object
    from datasets import Dataset
    return Dataset.from_dict(new_data)
