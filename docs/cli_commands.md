## CLI Usage:
```python
curator-evals --task <task_name> --model <model_identifier> --model-type <type>
```

### Required Arguments

**--task <task_name>** where <task_name> is evaluation task id to run and it can be:

- `math_correctness_evals`: Evaluate mathematical reasoning capabilities
- `code_correctness_evals`: Evaluate code generation and correctness

**--model <model_identifier>** where <model_identifier> is the model path on HuggingFace Hub:

- Model identifier, available on Huggingface, OpenAI, orTogether AI

**--model-type <type>** where <type> specifies the model architecture and it can be:

- `llm` - Standard language models using vLLM for efficient inference (e.g., GPT, LLaMA, Qwen).
- `llm_adapter` - Language models with adapters (LoRA, QLoRA, etc.) for fine-tuned models.

### Optional Arguments

**--adapter-name <adapter_identifier>** where <adapter_identifier> is the adapter path on HuggingFace Hub:

- `collinear-ai/coding_curator_python_classification_100425`

**--debug** enables debug mode for benchmarking on 100 examples.

**--tensor-parallel-size <num_gpus>** where <num_gpus> is the number of GPUs to use for tensor parallelism:

- Default: `1`
- Example: `-tensor-parallel-size 4`

**--use-server** enables vLLM server mode

**--server-url <url>** where <url> is the URL of the vLLM server:

- Default: `http://localhost:8000`

**--input-format <format>** where <format> specifies the input format for the model:

- `just_prompt` - Uses only the prompt field from the dataset
- `llama_math_correctness_prompt` - Formats input for Llama-based math correctness evaluation with question, solution, and response
- `phi_math_correctness_prompt` - Formats input for Phi-based math correctness evaluation with question, ground truth, and student attempt
- `code_correctness_prompt` - Formats input for code correctness evaluation with question and code output
- `llama3_instruction_following_judge` - Formats input for Llama3-based instruction following evaluation with prompt and response
- `coherence_llm_judge` - Formats input for coherence evaluation with prompt and response
- `instruction_complexity_llm_judge` - Formats input for instruction complexity evaluation with just the prompt

Each format uses specific Jinja2 templates to structure the input data appropriately for different types of evaluations (math correctness, code correctness, instruction following, coherence, and complexity assessment).

**--output-format <format>** where <format> specifies the output format (defaults to input-format if not specified):

- `collinear_llama3_judge` - Extracts output from Llama3 model JSON responses with "output" field
- `collinear_phi_judge` - Extracts output from Phi model JSON responses with "output" field
- `collinear_code_qwen_judge` - Extracts first digit after "[RESULT]" key from Code Qwen model responses
- `first_digit_after_output_key` - Extracts first digit after "output" key in the text
- `just_output` - Returns the raw output text without any parsing

Each format uses specific parsing logic to extract scores/predictions from different model response formats:

JSON-based formats look for structured responses with an "output" field
Key-based formats search for specific markers like "[RESULT]" or "output" followed by numeric values
The raw format preserves the original model output without modification

**--provider <provider_name>** where <provider_name> specifies the model provider:

- Default: `vllm`
- Options: `openai`, `togetherai`
- `-api-key <key>` where <key> is the API key for the model provider when using external APIs