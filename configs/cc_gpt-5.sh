#!/bin/bash
source .env
echo $OPENAI_API_KEY
# Configuration for evaluating gpt5 model on math correctness
curator-evals --task code_correctness \
  --model gpt-5 \
  --model-type llm \
  --use-server \
  --server-url None \
  --provider openai \
  --api-key $OPENAI_API_KEY \
  --debug \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge \
  --debug