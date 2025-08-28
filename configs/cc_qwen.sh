#!/bin/bash

# Configuration for evaluating Qwen model on code correctness

curator-evals --task code_correctness \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --model-type llm \
  --use-server \
  --server-url http://localhost:8000 \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge 