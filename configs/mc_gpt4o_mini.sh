#!/bin/bash
source .env
echo $OPENAI_API_KEY
# Configuration for evaluating gpt4o model on math correctness
curator-evals --task math_correctness \
  --model gpt-4o-mini-2024-07-18 \
  --model-type llm \
  --use-server \
  --server-url None \
  --provider openai \
  --api-key $OPENAI_API_KEY \
  --input-format llama_math_correctness_prompt \
  --output-format first_digit_after_output_key