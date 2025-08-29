#!/bin/bash

# Configuration for evaluating llama model on math correctness
curator-evals --task code_correctness \
  --model meta-llama/Llama-3.1-8B \
  --model-type llm \
  --use-server \
  --server-url http://localhost:8000 \
  --debug \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge