import json
import re
from typing import Dict, Any

def _first_digit_after_key(text: str, key_string: str = "[RESULT]") -> int:
    """Extract result from text using key string"""
    try:
        result_text = text.split(key_string)[1]
        # Find first digit after key string
        for char in result_text:
            if char.isdigit():
                return int(char)
    except:
        pass
    return 0

# def _extract_json_output(text: str) -> int:
#     """Extract output from JSON response"""
#     try:
#         # Find JSON object in text
#         json_match = re.search(r'\{[^}]*"output"[^}]*\}', text)
#         if json_match:
#             json_str = json_match.group(0)
#             data = json.loads(json_str)
#             return int(data.get("output", 0))
#     except:
#         pass
#     return 0

def _extract_json_output(text: str) -> int:
    """Extract output from JSON response"""
    # First try: regex to directly extract the "output" value
    try:
        output_match = re.search(r'"output"\s*:\s*(\d+)', text)
        if output_match:
            return int(output_match.group(1))
    except:
        pass
    
    # Fallback: try full JSON parsing
    try:
        cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
        data = json.loads(cleaned, strict=False)
        return int(data.get("output", 0))
    except:
        pass
    
    # Both methods failed
    print(f"Failed to extract output. Text preview: {text[:150]}")
    return 0

def _extract_phi_output(text: str) -> int:
    """Extract output from Phi model response"""
    return _extract_json_output(text)

def _extract_llama_output(text: str) -> int:
    """Extract output from Llama model response"""
    return _extract_json_output(text)

def _extract_code_qwen_output(text: str) -> int:
    """Extract output from Code Qwen model response"""
    return _first_digit_after_key(text, "[RESULT]")

def _just_output(text: str) -> int:
    """Extract output from model response"""
    return text

def format_outputs(output: Any, output_format: str) -> Dict[str, Any]:
    """
    Format model output based on the specified output format.
    
    Args:
        output: Raw model output
        output_format: Format specification for output parsing
        
    Returns:
        Formatted output with score and prediction
    """
    generated_text = output['output']['text']
    default_score = 0
    
    if output_format == "collinear_llama3_judge":
        score = _extract_llama_output(generated_text)
    elif output_format == "collinear_phi_judge":
        score = _extract_phi_output(generated_text)
    elif output_format == "collinear_code_qwen_judge":
        score = _extract_code_qwen_output(generated_text)
    elif output_format == "first_digit_after_output_key":
        score = _first_digit_after_key(generated_text, "output")
    elif output_format == "just_output":
        score = _just_output(generated_text)
    else:
        print(f"Unknown output format: {output_format}")
        score = default_score
    
    return {
        **output,
        "score": score,
        "prediction": score,
    } 