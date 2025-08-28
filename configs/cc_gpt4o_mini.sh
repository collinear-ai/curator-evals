#!/bin/bash
source .env
echo $OPENAI_API_KEY
# Configuration for evaluating gpt4o model with code correctness
curator-evals --task code_correctness \
  --model gpt-4o-mini-2024-07-18 \
  --model-type llm \
  --use-server \
  --server-url None \
  --provider openai \
  --api-key $OPENAI_API_KEY \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge 