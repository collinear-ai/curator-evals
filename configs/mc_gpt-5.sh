#!/bin/bash
source .env
echo $OPENAI_API_KEY
# Configuration for evaluating gpt5 model on math correctness
curator-evals --task math_correctness \
  --model gpt-5 \
  --model-type llm \
  --use-server \
  --server-url None \
  --provider openai \
  --api-key $OPENAI_API_KEY \
  --debug \
  --input-format llama_math_correctness_prompt \
  --output-format first_digit_after_output_key \
  --debug