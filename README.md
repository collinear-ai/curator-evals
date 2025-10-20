<h1 align="center">Curator Evals</h1>

<p align="center">
  A library for evaluating language models on various tasks using the Curator Eval Bench dataset
</p>

<p align="center">
  <a href="https://deepwiki.com/collinear-ai/curator-evals" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"/>
  </a>
  <a href="https://blog.collinear.ai/p/curator-evals">
    <img src="https://img.shields.io/badge/Blog-FF5733?style=for-the-badge&logoColor=white&labelColor=FF5733&color=FF5733" alt="Blog"/>
  </a>
</p>

<!--
# Curator Evals

A library for evaluating language models on various tasks using the Curator Eval Bench dataset -->

## 🎉 What's New

- [2025.10.20] Added Coherence evaluation metric using preference_ranking_agreement, with new input/output formats (`coherence_llm_judge`, `collinear_llama3_judge`).
- [2025.10.16] Introduced the preference_ranking_agreement() function, a new metric for evaluating alignment between model-generated preference scores and human-annotated rankings.
- [2025.10.02] Added Math Correctness evaluation metric with support for accuracy, precision, recall, F1, and new prompt options (`llama_math_correctness_prompt`, `phi_math_correctness_prompt`).
- [2025.09.04] Added support for Together.ai hosted models with asynchronous generation and built-in rate limiting for efficient concurrent requests.
- [2025.08.23] Improved OpenAI integration with asynchronous generation, concurrent request handling, and reasoning support for GPT-5 and o-series models.  
- [2025.08.19] Added vLLM integration with chat template support, asynchronous generation, and concurrent request handling for efficient completions. 


## Features

- **Task-Specific Evaluations** – Evaluate models on code and math correctness tasks using [Curator Eval Bench](https://huggingface.co/datasets/collinear-ai/curator_evals_bench) dataset.
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

<p style="text-align:center; font-size:14px; color:gray;">
Evaluated using <code>--task code_correctness</code> with 
<code>--input-format code_correctness_prompt</code> and 
<code>--output-format collinear_code_qwen_judge</code>.
</p>


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


<h2 style="text-align:center;">Math Correctness LeaderBoard</h2>

<p style="text-align:center; font-size:14px; color:gray;">
Evaluated using <code>--task math_correctness</code> with 
<code>--input-format phi_math_correctness_prompt</code> and 
<code>--output-format collinear_phi_judge</code>.
</p>

<div align="center">

<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Model</th>
      <th>Accuracy (%)</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Qwen3-8B</td>
      <td align="center"><b>93.95</b></td>
      <td align="center">0.968</td>
      <td align="center">0.970</td>
      <td align="center">0.969</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Qwen2.5-Coder-7B-Instruct</td>
      <td align="center">93.90</td>
      <td align="center">0.969</td>
      <td align="center">0.968</td>
      <td align="center">0.968</td>
    </tr>
    <tr>
      <td>3</td>
      <td>gemma-3-12b-it</td>
      <td align="center">93.75</td>
      <td align="center">0.968</td>
      <td align="center">0.967</td>
      <td align="center">0.968</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Seed-Coder-8B-Instruct</td>
      <td align="center">87.20</td>
      <td align="center">0.967</td>
      <td align="center">0.898</td>
      <td align="center">0.931</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Qwen2.5-Coder-3B-Instruct</td>
      <td align="center">86.30</td>
      <td align="center">0.966</td>
      <td align="center">0.889</td>
      <td align="center">0.926</td>
    </tr>
    <tr>
      <td>6</td>
      <td>DeepSeek-R1-0528-Qwen3-8B</td>
      <td align="center">76.00</td>
      <td align="center">0.967</td>
      <td align="center">0.779</td>
      <td align="center">0.863</td>
    </tr>
  </tbody>
</table>

</div>


<h2 style="text-align:center;">Coherence Evaluation Leaderboard</h2>

<p style="text-align:center; font-size:14px; color:gray;">
Evaluated using <code>--task coherence</code> with 
<code>--input-format coherence_llm_judge</code> and 
<code>--output-format collinear_llama3_judge</code>.
</p>

<div align="center">

<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Model</th>
      <th>Preference Ranking Agreement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>gemma-3-12b-it</td>
      <td align="center"><b>0.8189</b></td>
    </tr>
    <tr>
      <td>2</td>
      <td>Qwen2.5-Coder-3B-Instruct</td>
      <td align="center">0.8052</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Qwen2.5-Coder-7B-Instruct</td>
      <td align="center">0.7866</td>
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
