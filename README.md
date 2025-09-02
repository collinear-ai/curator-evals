# Curator Evals

A library for evaluating language models on various tasks using the Curator Eval Bench dataset.

## Features

- **Task-Specific Evaluations** – Evaluate models on code using Curator Eval Bench dataset.
- **Flexible Model Support** – Works with LLMs on huggingface, togetherai, and openai.
- **Detailed Metrics** – Provides accuracy, coherence scores, complexity ratings, and component breakdowns.
- **Command-Line and Python API** – Run quick CLI commands or integrate programmatically in your workflow.

## Setup

```bash
conda create -n curator python=3.11 -y

conda activate curator

git clone https://github.com/collinear-ai/curator-evals.git

cd curator-evals

pip install uv

uv pip install -e .
```
## Basic Example
Run vllm server in one terminal.
```bash
python -u \
    -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model Qwen/Qwen2.5-Coder-3B-Instruct
```
Start code correctness benchmark on second terminal.
```bash
curator-evals --task code_correctness \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --model-type llm \
  --use-server \
  --server-url http://localhost:8000 \
  --input-format code_correctness_prompt \
  --output-format collinear_code_qwen_judge
```
You can find more examples in `configs` folder.

<!-- ## Code Correctness LeaderBoard
| Rank | Model                          | Accuracy (%) |
|:---:|:-------------------------------:|:-------:|
| 1   | Qwen2.5-Coder-7B-Instruct       | **76.88** |
| 2   | Seed-Coder-8B-Instruct          | 71.27 |
| 3   | gpt-4o                          | 63.74 |
| 4   | DeepSeek-R1-0528-Qwen3-8B       | 63.67 |
| 5   | Qwen3-8B                        | 60.59 |
| 6   | Qwen2.5-Coder-3B-Instruct       | 46.77 | -->
<h2 style="text-align:center;">Code Correctness LeaderBoard</h2>

<div align="center">

<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Model</th>
      <th>Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Qwen2.5-Coder-7B-Instruct</td>
      <td align="center"><b>76.88</b></td>
    </tr>
    <tr>
      <td>2</td>
      <td>Seed-Coder-8B-Instruct</td>
      <td align="center">71.27</td>
    </tr>
    <tr>
      <td>3</td>
      <td>gpt-4o</td>
      <td align="center">63.74</td>
    </tr>
    <tr>
      <td>4</td>
      <td>DeepSeek-R1-0528-Qwen3-8B</td>
      <td align="center">63.67</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Qwen3-8B</td>
      <td align="center">60.59</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Qwen2.5-Coder-3B-Instruct</td>
      <td align="center">46.77</td>
    </tr>
  </tbody>
</table>

</div>


## Benchmarking Details

The evaluation dataset is hosted on HuggingFace Hub at `collinear-ai/curator_evals_bench`. Each task is a subset of the dataset, containing different splits for various dataset sources.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you find Curator Evals useful, do not forget to cite us!
```
@misc{curator-evals,
  author       = {Mackey, Tsach and Shafique, Muhammad Ali and Kumar, Anand},
  title        = {Curator Evals: A Benchmark for High-quality Post-training Data Curation},
  year         = {2025},
  month        = {Sep},
  howpublished = {\url{https://github.com/collinear-ai/curator-evals}}
}
```
