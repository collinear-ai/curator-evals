from typing import Dict, List, Any
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

def get_metrics(results: List[Dict[str, Any]], task: str) -> Dict[str, float]:
    """
    Calculate metrics based on the evaluation results and task type.
    
    Args:
        results: List of model outputs
        task: Name of the evaluation task
    
    Returns:
        Dictionary containing the calculated metrics
    """
    if task == "math_correctness":
        return _calculate_math_metrics(results)
    elif task == "instruction_following":
        return _calculate_instruction_following_metrics(results)
    elif task == "coherence":
        return _calculate_coherence_metrics(results)
    elif task == "instruction_complexity":
        return _calculate_complexity_metrics(results)
    elif task == "code_correctness":
        return _calculate_code_metrics(results)
    elif task == "quality_of_reasoning":
        return _calculate_quality_of_reasoning_metrics(results)
    else:
        raise ValueError(f"Unknown task type: {task}")

def _calculate_math_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for math correctness evaluation"""
    scores = [r.get("prediction", r.get("score", 0)) > 0 for r in results]
    labels = [r["label"] for r in results]
    accuracy = np.mean([s == l for s, l in zip(scores, labels)])
    precision = precision_score(labels, scores)
    recall = recall_score(labels, scores)
    f1 = f1_score(labels, scores)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def _calculate_instruction_following_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for instruction following evaluation"""
    # benchmark_source to data 
    benchmark_source_to_data = defaultdict(list)
    for result in results:
        benchmark_source_to_data[result['benchmark_source']].append(result)
    
    all_metrics = {}
    # get the preference scores using the preference_ranking_agreement function
    for benchmark_source, data in benchmark_source_to_data.items():
        prompts = [r["prompt"] for r in data]
        labels = [float(r["label"]) for r in data]
        scores = [float(r.get("prediction", r.get("score", 0.0))) for r in data]
        metrics = preference_ranking_agreement(prompts, labels, scores)
        all_metrics[f"{benchmark_source}_preference_ranking_agreement"] = metrics["accuracy"]
    
    return all_metrics

def _calculate_quality_of_reasoning_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for quality of reasoning evaluation"""
    scores = [r.get("prediction", r.get("score", 0.0)) for r in results]
    human_scores = [int(r.get("label", 0.0)) for r in results]

    #convert the scores (-inf, inf) into a binary classification using a threshold of 0
    binary_scores = [1 if s > 0 else 0 for s in scores]
    accuracy = np.mean([s == h for s, h in zip(binary_scores, human_scores)])

    f1 = f1_score(human_scores, binary_scores)
    precision = precision_score(human_scores, binary_scores)
    recall = recall_score(human_scores, binary_scores)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision, 
        "recall": recall,
    }

def _calculate_coherence_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for coherence evaluation"""

    # Extract prompts, labels, and scores from results
    prompts = [r["prompt"] for r in results]
    labels = [float(r["label"]) for r in results]

    ## check if prediction or score is present, else raise an error
    for r in results:
        if "prediction" not in r and "score" not in r:
            raise ValueError("Either prediction or score must be present in the results")
    scores = [float(r.get("prediction", r.get("score"))) for r in results]
    
    #get the preference scores using the preference_ranking_agreement function
    preference_metrics = preference_ranking_agreement(prompts, labels, scores)
    
    return {
        "preference_ranking_agreement": preference_metrics["accuracy"]
    }

def _calculate_complexity_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for instruction complexity evaluation"""
    complexity_scores = [float(r.get("prediction", 0.0)) for r in results]
    gt_complexity_scores = [float(r.get("label", 0.0)) for r in results]
    #get the correlation between the complexity scores and the gt complexity scores

    pearson_corr, _ = pearsonr(complexity_scores, gt_complexity_scores)
    spearman_corr, _ = spearmanr(complexity_scores, gt_complexity_scores)
    
    return {
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr
    }

def _calculate_code_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for code correctness evaluation"""
    scores = [int(r.get("prediction", r.get("score", 0.0))) for r in results]
    labels = [int(r.get("label", 0.0)) for r in results]
    accuracy = np.mean([s == l for s, l in zip(scores, labels)])
    
    return {
        "accuracy": accuracy,
    } 

def preference_ranking_agreement(prompts: List[str], labels: List[float], scores: List[float]) -> Dict[str, Any]:
    """
    Calculate preference ranking agreement between model predictions and ground truth labels.
    
    This function evaluates how well a model's preference scores align with human preference
    labels when ranking multiple responses to the same prompt. It groups responses by prompt
    and checks if the model correctly identifies the response with the highest human preference
    score.
    
    Args:
        prompts: List of prompt strings
        labels: List of ground truth preference scores (floats)
        scores: List of model predicted preference scores (floats)
    
    Returns:
        Dict: Dictionary containing:
            - 'total_groups': Number of prompt groups with 2+ responses (int)
            - 'correct_groups': Number of groups where model correctly identified best response (int)
            - 'accuracy': Fraction of groups where model was correct (float or None)
            - 'baseline': Expected accuracy from random guessing (float or None)
    """
    # Group by prompt
    prompt_groups = defaultdict(list)
    for prompt, label, score in zip(prompts, labels, scores):
        prompt_groups[prompt].append({
            'label': label,
            'score': score
        })
    
    total_groups = 0
    correct_groups = 0
    baseline = 0
    for entries in prompt_groups.values():
        if len(entries) < 2:
            continue
        # if the responses have the same label, skip it 
        if len(set(entry['label'] for entry in entries)) == 1:
            continue           
        total_groups += 1

        # Find the response with the highest label
        best_label_entry = max(entries, key=lambda x: x['label'])
        
        # Find the response with the highest score
        best_score_entry = max(entries, key=lambda x: x['score'])
        
        # Check if the model correctly identified the best response
        if best_label_entry == best_score_entry:
            correct_groups += 1

        # Calculate baseline random accuracy
        baseline += 1 / len(entries)
    
    accuracy = correct_groups / total_groups if total_groups > 0 else None
    baseline = baseline / total_groups if total_groups > 0 else None
    return {
        'total_groups': total_groups,
        'correct_groups': correct_groups,
        'accuracy': accuracy,
        'baseline': baseline
    }