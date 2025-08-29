#!/bin/bash

# Configuration for evaluating Phi model on math correctness
curator-evals --task code_correctness \
  --model microsoft/Phi-3.5-mini-instruct \
  --model-type llm \
  --use-server \
  --server-url http://localhost:8000 \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge