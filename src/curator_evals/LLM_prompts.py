from jinja2 import Template
from typing import Dict, Any
code_correctness_prompt = Template("""You are a helpful assistant tasked with evaluating the correctness of a code output given an input question.

## Evaluation Criteria:
1. The code output is **correct** if it:
   - Is functionally accurate (produces the expected result),
   - Is syntactically correct (free of syntax errors),
   - Follows standard coding conventions,
   - Fully satisfies the requirements stated in the input question.

2. Return **1** if the code output is correct, and **0** if it is incorrect.

Respond in the following format only:  
   [RESULT] <1 or 0>  
   Do not include any explanations or additional text.

## Input Question:
{{prompt}}

## Code Output to Evaluate:
{{response}}
""")

phi_math_correctness_prompt = Template("""You are an AI assistant for grading math problems. The user will provide you with the question itself, an attempt made by
a student and the ground truth answer to the problem. Your job is to judge whether the student attempt is correct by comparing it with the
correct answer.

You must:
1. Independently solve the math question from scratch.
2. Evaluate the provided student attempt by comparing with the ground truth answer.
3. Confirm that the final answer in the student answer exactly matches the final answer of the ground truth solution.
5. If every step in student attempt is correct and the final answer is exactly the same as ground truth answer, output 1, otherwise output 0.
6. Finally, Your output must be a valid JSON object containing:
 • "output": An integer (1 or 0) representing the classification.

Input details:

## Question:
{{prompt}}

## Ground Truth Answer:
{{solution}}

## Student Attempt:
{{response}}

Now, please output the following as a JSON object:

{ 
"output": <classification score (0 or 1)> (int datatype),
}""")

llama_math_correctness_prompt = Template("""You will classify whether the math answer is correct or not based on the given question and solution.

Follow these steps:

1. Assess the student's answer against the correct solution
2. Classify it as 0 (incorrect) or 1 (correct)
3. Provide Rationale: Explain your classification decision with a brief rationale.
4. Finally, output the results as a JSON object with the fields "rationale" and "output" where "output" contains the classification (0 or 1)

# Question:
{{prompt}}

# Correct Solution:
{{solution}}

# Student Answer:
{{response}}

Now, please output the following as a JSON object:

{
"rationale": <verbal feedback> (str datatype),
"output": <classification score (0 or 1)> (int datatype),
}""")

llama_instruction_following_prompt = Template("""
You are an AI assistant for evaluating whether a response followed the instructions specified in a prompt. 
You will be given a prompt and a response. 

# Prompt:
{{prompt}}

# Response:
{{response}}

Now, please output the following as a JSON object:

{
"rationale": <verbal feedback> (str datatype),
"output": <classification score (0 or 1)> (int datatype),
}
""")
      

coherence_llm_judge_prompt = Template("""
You are an AI assistant for evaluating the coherence of a response given an input question.
You will be given a prompt and a response. Please provide a score between 0 (not coherent) and 1 (coherent).

# Prompt:
{{prompt}}

# Response:
{{response}}

Now, please output the following as a JSON object:

{
"rationale": <verbal feedback> (str datatype),
"output": <classification score (0 or 1)> (int datatype),
}"""
)                                    

instruction_complexity_llm_judge_prompt =Template("""
You are an AI assistant for evaluating the complexity of an instruction.
Please provide a score between 0 and 5 based on the complexity of the instruction.

# Instruction:
{{prompt}}

Now, please output the following as a JSON object:

{
"rationale": <verbal feedback> (str datatype),
"output": <classification score (0 or 5)> (int datatype),
}"""
)        
                                      

def _format_collinear_math_llama3_judge(row):
    return llama_math_correctness_prompt.render(
        prompt=row["prompt"],
        solution=row["solution"],
        response=row["response"]
    )

def _format_collinear_math_phi_judge(row):
    return phi_math_correctness_prompt.render(
        prompt=row["prompt"],
        solution=row["solution"],
        response=row["response"]
    )

def _format_collinear_code_qwen_judge(row):
    return code_correctness_prompt.render(
        prompt=row["prompt"],
        response=row["response"]
    )

def _format_instruction_following_llama3_judge(row):
    return llama_instruction_following_prompt.render(
        prompt=row["prompt"],
        response=row["response"]
    )

def _format_coherence_llm_judge(row):
    return coherence_llm_judge_prompt.render(
        prompt=row["prompt"],
        response=row["response"]
    )

def _format_instruction_complexity_llm_judge(row):
    return instruction_complexity_llm_judge_prompt.render(
        prompt=row["prompt"]
    )

def _just_prompt(row): 
  return row["prompt"]

def format_inputs(row: Dict[str, Any], input_format: str) -> str:
    if input_format == "just_prompt":
      return _just_prompt(row)
    elif input_format == "llama_math_correctness_prompt": 
      return _format_collinear_math_llama3_judge(row)
    elif input_format == "phi_math_correctness_prompt":
      return _format_collinear_math_phi_judge(row)
    elif input_format == "code_correctness_prompt":
      return _format_collinear_code_qwen_judge(row)
    elif input_format == "llama3_instruction_following_judge":
      return _format_instruction_following_llama3_judge(row)
    elif input_format == "coherence_llm_judge":
      return _format_coherence_llm_judge(row)
    elif input_format == "instruction_complexity_llm_judge":
      return _format_instruction_complexity_llm_judge(row)
    else:
      raise ValueError(f"Invalid input_format: {input_format}")