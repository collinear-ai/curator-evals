from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Literal
# from LiveCodeBench.lcb_runner import prompts
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
try:
    # Check if we're in a Jupyter notebook environment
    get_ipython()
    # If we get here, we're in a notebook, so don't import peft
    PeftModel = None
except NameError:
    # Not in a notebook, safe to import
    from peft import PeftModel
from .LLM_prompts import format_inputs
from .output_formatters import format_outputs
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import tempfile
import shutil
import hashlib
import aiohttp
import json
import asyncio
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
import re

from vllm import SamplingParams
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
from together import AsyncTogether

class AsyncApiServer:
    """
    Unified class to handle vLLM server or OpenAI API calls through the same interface.
    Defaults to vLLM server.
    """
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        provider: Literal["vllm", "openai"] = "vllm",
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-4o"
    ):
        self.server_url = server_url
        self.provider = provider
        self.api_key = api_key
        self.model = model or "gpt-4o"
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            stop=None,
        )

    async def generate(self, prompts: List[str]) -> List[Any]:
        if self.provider == "vllm":
            return await self._generate_vllm(prompts)
        elif self.provider == "openai":
            return await self._generate_openai(prompts)
        elif self.provider == "togetherai":
            return await self._generate_togetherai(prompts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_vllm(self, prompts: List[str]) -> List[Any]:
        
        # Load tokenizer once for chat template
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        
        async def fetch(prompt):
            try:
                # Apply chat template
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_url}/v1/completions",
                        json={
                            "prompt": formatted_prompt,  # Use formatted prompt
                            "max_tokens": self.sampling_params.max_tokens,
                            "temperature": self.sampling_params.temperature,
                            "top_p": self.sampling_params.top_p,
                            "stop": self.sampling_params.stop, #[tokenizer.eos_token, "<|im_end|>"],  # Add proper stop tokens
                        }
                    ) as response:
                        result = await response.json()
                        return {"formatted_prompt": formatted_prompt, "text": result["choices"][0]["text"]}
            except Exception as e:
                print(f"Request failed: {e}")
                return {"formatted_prompt": formatted_prompt, "text": ""}
        
        # Process requests concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)
        
        async def limited_fetch(prompt):
            async with semaphore:
                await asyncio.sleep(0.1)
                return await fetch(prompt)
        
        tasks = [limited_fetch(prompt) for prompt in prompts]
        results = []
        for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses", unit="prompt"):
            results.append(await result)
        
        return results        
    
    async def _generate_togetherai(self, prompts: List[str]) -> List[Any]:
        
        client = AsyncTogether(api_key=self.api_key)
        
        async def fetch(prompt):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.sampling_params.max_tokens,
                    temperature=self.sampling_params.temperature,
                    top_p=self.sampling_params.top_p,
                )
                text = response.choices[0].message.content
                return {"text": text}
            except Exception as e:
                print(f"Request failed: {e}")
                return {"text": ""}
        
        # Process requests concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def limited_fetch(prompt):
            async with semaphore:
                await asyncio.sleep(0.1)  # Small delay
                return await fetch(prompt)
        
        # Use limited_fetch instead of fetch
        tasks = [limited_fetch(prompt) for prompt in prompts]
        results = []
        for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses", unit="prompt"):
            results.append(await result)
        
        return results
    
    async def _generate_openai(self, prompts: List[str]) -> List[Any]:
        client = AsyncOpenAI(api_key=self.api_key)
        results = []

        # OpenAI API supports batch requests via 'messages' parameter, but not true batch for chat completions.
        # We'll send requests concurrently using asyncio.gather for efficiency.
        async def fetch(prompt):
            # Check if model supports reasoning (gpt-5 or o-series)
            supports_reasoning = (
                self.model.startswith("gpt-5") or 
                self.model.startswith("o1") or 
                self.model.startswith("o3")
            )
            
            if supports_reasoning:
                # Use responses.create for reasoning models
                response = await client.responses.create(
                    model=self.model,
                    input=prompt,
                    reasoning={"effort": "high"}
                )
                text = getattr(response, 'output_text', response.output)
            else:
                # Use regular chat completions for other models
                response = await client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.sampling_params.temperature,
                    top_p=self.sampling_params.top_p,
                )
                text = getattr(response, 'output_text', response.output)
            
            return {"text": text}

        # Process requests concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def limited_fetch(prompt):
            async with semaphore:
                await asyncio.sleep(0.1)  # Small delay
                return await fetch(prompt)

        tasks = [limited_fetch(prompt) for prompt in prompts]
        for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses", unit="prompt"):
            results.append(await result)
        return results


class ModelFactory:
    """Factory class for creating model instances based on model type"""
    
    @staticmethod
    def create_model(model_info: Dict[str, Any]) -> 'Model':
        """
        Create and return the appropriate model instance based on model type.
        
        Args:
            model_info: Dictionary containing model configuration
            
        Returns:
            An instance of the appropriate Model subclass
        """
        model_type = model_info.get("model_type", "").lower()
        
        if model_type == "llm" or model_type == "llm_adapter":
            return LLMModel(model_info)
        elif model_type == "classifier":
            return ClassifierModel(model_info)
        elif model_type == "ensemble":
            return EnsembleModel(model_info)
        elif model_type == "nvidia_complexity":
            return NvidiaComplexityModel(model_info)
        elif model_type == "regex_math":
            return RegexMathModel(model_info)
        elif model_type == "length_heuristic":
            return LengthHeuristic(model_info)
        elif model_type == "ppl_heuristic":
            return PPLHeuristic(model_info)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class Model(ABC):
    """Base class for all model types"""
    
    def __init__(self, model_info: Dict[str, Any]):
        self.model_info = model_info
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model and tokenizer based on model_info"""
        pass
    
    @abstractmethod
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the input data for the specific model type"""
        pass
    
    @abstractmethod
    def format_output(self, output: Any) -> Dict[str, Any]:
        """Format the model output into a standardized format"""
        pass
    
    async def inference(self, input_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference on a batch of inputs"""
        formatted_inputs = [self.format_input(input_data) for input_data in input_batch]
        outputs = await self._run_inference(formatted_inputs)
        return [self.format_output(output) for output in outputs]
    
    @abstractmethod
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[Any]:
        """Run the actual model inference"""
        pass
    
    async def evaluate(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Evaluate the model on the entire dataset"""
        results = []
        batch_size = 100  # Can be made configurable
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
            batch_results = await self.inference(batch)
            results.extend(batch_results)
        
        return results

class LLMModel(Model):
    """Implementation for LLM models using vLLM for efficient inference"""
    
    def _load_model(self):            
        # For server mode, just initialize the client
        if self.model_info.get("use_server", False):
            self.model = AsyncApiServer(
                self.model_info.get("server_url", "http://localhost:8000"),
                self.model_info.get("provider", "vllm"),
                self.model_info.get("api_key", None),
                self.model_info.get("model", "gpt-4o")
            )
            return
    
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_id"])
            
        # For direct vLLM usage, initialize the model
        print("Initializing vLLM with model...")
        self.model = LLM(
            model=self.model_info["model_id"],
            tensor_parallel_size=self.model_info.get("tensor_parallel_size", 1),
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
    
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        input_format = self.model_info["input_format"]
        prompt = format_inputs(input_data, input_format)
        return {
            **input_data,
            "text": prompt,
        }
    
    def format_output(self, output: Any) -> Dict[str, Any]:
        output_format = self.model_info.get("output_format", self.model_info.get("input_format"))
        return format_outputs(output, output_format)
    
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[Any]:
        # Use vLLM for inference
        prompts = [input_data["text"] for input_data in formatted_inputs]
        
        if isinstance(self.model, AsyncApiServer):
            outputs = await self.model.generate(prompts)
        else:
            # For direct vLLM usage, we need to run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                lambda: self.model.generate(prompts, self.sampling_params)
            )
        
        results = []
        for input_data, output in zip(formatted_inputs, outputs):
            results.append({
                **input_data,
                "output": output
            })
        return results

class ClassifierModel(Model):
    """Implementation for classifiers"""

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_id"])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_info["model_id"])
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data["prompt"]
        response = input_data["response"]
        # Format input for reward models
        text = f"user: {prompt} assistant: {response}"
        return {
            **input_data,
            "text": text
        }
    
    # def format_output(self, output: torch.Tensor) -> Dict[str, Any]:
    #     # Format reward model output
    #     score = output['output'].item()
    #     return {
    #         **output,
    #         "score": score,
    #         "prediction": score
    #     }
    def format_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # Format reward model output
        tensor_output = output['output']
        if isinstance(tensor_output, torch.Tensor):
            score = tensor_output.item()
        else:
            score = tensor_output
        
        # Convert tensors to JSON-serializable format while preserving original structure
        serializable_output = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                # Convert tensors to Python scalars or lists
                if v.numel() == 1:
                    serializable_output[k] = v.item()
                else:
                    serializable_output[k] = v.tolist()
            else:
                serializable_output[k] = v
        
        return {
            **serializable_output,  # This preserves all original keys with tensor-safe values
            "score": score,
            "prediction": score
        }
    
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        # Extract all texts from the batch
        texts = [input_data["text"] for input_data in formatted_inputs]
        
        # Tokenize all texts at once
        encoded = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        results = []
        with torch.no_grad():
            # Run inference on the entire batch
            outputs = self.model(**encoded)
            
            # Process each output
            for i, input_data in enumerate(formatted_inputs):
                # Extract the logits for this sample
                sample_logits = outputs.logits[i] if outputs.logits.dim() > 1 else outputs.logits
                
                results.append({
                    **input_data,
                    "output": sample_logits
                })
        
        return results


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MulticlassHead(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class CustomNvidiaModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(CustomNvidiaModel, self).__init__()

        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]

        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)

        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            task_type = {}

            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]

            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1

            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]

            return (task_type_1, task_type_2, task_type_prob)

        else:
            preds = torch.softmax(preds, dim=1)

            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]

            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

    def process_logits(self, logits):
        result = {}
        complexity_result = {}

        # Round 1: "task_type"
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        result["task_type_1"] = task_type_results[0]
        result["task_type_2"] = task_type_results[1]
        result["task_type_prob"] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1]
        target = "creativity_scope"
        result[target] = self.compute_results(creativity_scope_logits, target=target)

        # Round 3: "reasoning"
        reasoning_logits = logits[2]
        target = "reasoning"
        result[target] = self.compute_results(reasoning_logits, target=target)

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3]
        target = "contextual_knowledge"
        result[target] = self.compute_results(
            contextual_knowledge_logits, target=target
        )

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4]
        target = "number_of_few_shots"
        result[target] = self.compute_results(number_of_few_shots_logits, target=target)

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5]
        target = "domain_knowledge"
        result[target] = self.compute_results(domain_knowledge_logits, target=target)

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6]
        target = "no_label_reason"
        result[target] = self.compute_results(no_label_reason_logits, target=target)

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7]
        target = "constraint_ct"
        result[target] = self.compute_results(constraint_ct_logits, target=target)

        # Round 9: "prompt_complexity_score"
        complexity_result["prompt_complexity_score"] = [
            round(
                0.35 * creativity
                + 0.25 * reasoning
                + 0.15 * constraint
                + 0.15 * domain_knowledge
                + 0.05 * contextual_knowledge
                + 0.05 * few_shots,
                5,
            )
            for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
            )
        ]

        return complexity_result

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)

        logits = [
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        ]

        return self.process_logits(logits)

class NvidiaComplexityModel(Model):
    """Implementation for NVIDIA instruction complexity classifier"""
    
    def _load_model(self):
        # Load the NVIDIA complexity classifier
        config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
        
        self.model = CustomNvidiaModel(
            target_sizes=config.target_sizes,
            task_type_map=config.task_type_map,
            weights_map=config.weights_map,
            divisor_map=config.divisor_map,
        ).from_pretrained("nvidia/prompt-task-and-complexity-classifier")
        self.model.eval()
    
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # For complexity evaluation, we only need the prompt
        prompt = input_data["prompt"]
        # Format as expected by the NVIDIA model
        text = f"Prompt: {prompt}"
        
        # Tokenize the input
        encoded = self.tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        
        return {
            **input_data,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def format_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # Extract the complexity score and other metrics
        complexity_score = output['output'].get("prompt_complexity_score", [0.0])[0]
        
        return {
            **output,
            "prediction": complexity_score,
        }
    
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        with torch.no_grad():
            for input_data in formatted_inputs:
                # Create batch for the model
                batch = {
                    "input_ids": input_data["input_ids"],
                    "attention_mask": input_data["attention_mask"]
                }
                
                # Run inference
                output = self.model(batch)
                
                results.append({
                    **input_data,
                    "output": output
                })
        return results 

class RegexMathModel(Model):
    """Implementation for regex-based math correctness detection"""
    
    def _load_model(self):
        # No model loading needed for regex matching
        self.model = None
        self.tokenizer = None
    
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract prompt, response, and solution from input
        prompt = input_data.get("prompt", "")
        response = input_data.get("response", "")
        solution = input_data.get("solution", "")
        
        return {
            **input_data,
            "prompt": prompt,
            "response": response,
            "solution": solution
        }
    
    def format_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # Return the comparison result
        return {
            **output,
            "score": output.get("score", 0),
            "response_boxed": output.get("response_boxed", ""),
            "solution_boxed": output.get("solution_boxed", ""),
            "prediction": output.get("score", 0),
        }
    
    def _extract_boxed_value(self, text: str) -> str:
        """Extract the value inside \\boxed{} from text"""
        # Pattern to match \boxed{...} where ... can contain any characters
        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return ""
    
    def _normalize_math_expression(self, expr: str) -> str:
        """Normalize math expression for comparison"""
        # Remove extra whitespace
        expr = re.sub(r'\s+', ' ', expr.strip())
        # Remove leading/trailing whitespace
        expr = expr.strip()
        return expr
    
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for input_data in formatted_inputs:
            response = input_data["response"]
            solution = input_data["solution"]
            
            # Extract boxed values
            response_boxed = self._extract_boxed_value(response)
            solution_boxed = self._extract_boxed_value(solution)
            
            # Normalize expressions
            response_normalized = self._normalize_math_expression(response_boxed)
            solution_normalized = self._normalize_math_expression(solution_boxed)
            
            # Compare and assign score
            if response_normalized == solution_normalized:
                score = 1
            else:
                score = 0
            
            results.append({
                **input_data,
                "score": score,
                "response_boxed": response_boxed,
                "solution_boxed": solution_boxed,
                "prediction": score,
            })
        
        return results 

class LengthHeuristic(Model):
    """Implementation for length heuristic"""
    
    def _load_model(self):
        # No model loading needed for length calculation
        self.model = None
        self.tokenizer = None
    
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract prompt from input
        prompt = input_data.get("prompt", "")
        
        return {
            **input_data,
            "prompt": prompt
        }
    
    def format_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # Return the length as score
        return {
            **output,
            "score": output.get("score", 0),
            "prediction": output.get("score", 0),
        }
    
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for input_data in formatted_inputs:
            prompt = input_data["prompt"]
            
            # Calculate prompt length as score
            score = len(prompt)
            
            results.append({
                **input_data,
                "score": score,
                "prediction": score,
            })
        
        return results 

class PPLHeuristic(Model):
    """Implementation for perplexity-based response quality evaluation"""
    
    def _load_model(self):
        # Load a language model for perplexity calculation
        model_name = self.model_info.get("model_id")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def format_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract response text for perplexity calculation
        if self.model_info['task'] == "coherence":
            text = input_data.get("response", "")
        elif self.model_info['task'] == "instruction_complexity":
            text = input_data.get("prompt", "")
        else: 
            text = "user: " + input_data.get("prompt", "") + "\nassistant: " + input_data.get("response", "")
        
        return {
            **input_data,
            "text": text
        }
    
    def format_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # Return the perplexity as prediction (lower is better, so we might want to invert)
        ppl = output.get("perplexity", 0.5)
        # Invert perplexity so higher values are better (more natural text)
        # Add small epsilon to avoid division by zero
        score = 1.0 / (ppl + 1e-8)
        
        return {
            **output,
            "perplexity": ppl,
            "prediction": score,
            "score": score
        }
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of given text"""
        if not text.strip():
            return float('inf')
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            
            # Calculate perplexity
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

            #if the perplexity is nan, it's likely a very shot string, replace with a very high ppl 
            if torch.isnan(torch.exp(loss)):
                perplexity = 2
            
        return perplexity
    
    async def _run_inference(self, formatted_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for input_data in formatted_inputs:
            text = input_data["text"]
            
            # Calculate perplexity
            perplexity = self._calculate_perplexity(text)
            

            results.append({
                **input_data,
                "perplexity": perplexity
            })
        
        return results 