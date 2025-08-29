#!/bin/bash
source .env
echo $TOGETHER_API_KEY
# Configuration for evaluating Meta-Llama-3.1-8B model using together ai on math correctness
curator-evals --task code_correctness \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --model-type llm \
  --use-server \
  --server-url None \
  --provider togetherai \
  --api-key $TOGETHER_API_KEY \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge \
  --debug