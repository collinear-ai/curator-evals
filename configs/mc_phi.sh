#!/bin/bash

# Configuration for evaluating Phi model on math correctness
curator-evals --task math_correctness \
  --model microsoft/Phi-3.5-mini-instruct \
  --model-type llm \
  --use-server \
  --server-url http://localhost:8000 \
  --input-format phi_math_correctness_prompt \
  --output-format collinear_phi_judge