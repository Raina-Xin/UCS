# main.py
"""
This script implements the UCS method for demonstration selection in In-Context Learning.
"""
import os
# Disable tokenizer parallelism to avoid warnings when sklearn forks processes
# This is safe because we process embeddings sequentially and use GPU parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import uuid  # For generating unique run IDs
import sys
import time
import requests
import json
import random
import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import numpy as np
import logging
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, DatasetDict

# HuggingFace Hub for authentication
try:
    from huggingface_hub import login
except ImportError:
    login = None
    logging.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")

# Databricks SDK
try:
    from databricks.sdk import WorkspaceClient
except ImportError:
    raise ImportError("databricks-sdk not installed. Install with: pip install databricks-sdk")

# Add OpenICL to path if not installed as package
try:
    from openicl import DatasetReader
    from openicl.icl_retriever import DPPRetriever, MDLRetriever, VotekRetriever, VotekSGTRetriever, DPPSGTRetriever, MDLSGTRetriever
except ImportError:
    openicl_path = Path(__file__).parent.parent / "OpenICL"
    if openicl_path.exists():
        sys.path.insert(0, str(openicl_path))
        from openicl import DatasetReader
        from openicl.icl_retriever import DPPRetriever, MDLRetriever, VotekRetriever
        from openicl.icl_retriever import VotekSGTRetriever, DPPSGTRetriever, MDLSGTRetriever
    else:
        raise ImportError(f"OpenICL not found. Please install it or ensure it's at {openicl_path}")

from data_utils import (
    load_banking77, load_banking77_label_names,
    load_clinc150, load_clinc150_label_names,
    load_hwu64, load_hwu64_label_names,
    load_bbeh_task,
)
from embed_and_cluster import run_dbscan_thresholded, get_or_compute_embeddings
from dict_knowledge import fit_dictionary_knowledge, DictKnowledgeConfig, atom_frequencies
from icl_eval import (
    build_demo_block_banking77, build_prompt_banking77,
    build_demo_block_clinc150, build_prompt_clinc150,
    build_demo_block_hwu64, build_prompt_hwu64,
    build_demo_block_bbh, build_prompt_bbh, evaluate_bbh_answer,
    clean_generation, best_label_match
)
from tqdm import tqdm
from plot import plot_aggregate_cluster_statistics, plot_cluster_distribution


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "compare_databricks_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")



def query_dbx_chat(
    endpoint: str,
    user_text: str,
    system_text: str = "You are helpful assistant.",
    temperature: float = 0.1,
    max_tokens: int = 64,
    timeout: int = 120,
    max_retries: int = 8,
    min_interval_s: float = 0.20,   # throttle: at most 4 req/s per process
    _state: dict = {"last_call": 0.0},
) -> str:
    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    token = os.environ["DATABRICKS_TOKEN"]
    url = f"{host}/serving-endpoints/{endpoint}/invocations"

    # ---- simple per-process throttle ----
    now = time.time()
    sleep_for = _state["last_call"] + min_interval_s - now
    if sleep_for > 0:
        time.sleep(sleep_for)
    _state["last_call"] = time.time()

    payload = {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if r.status_code == 429:
                # Respect Retry-After if provided, else exponential backoff + jitter
                ra = r.headers.get("Retry-After")
                if ra is not None:
                    delay = float(ra)
                else:
                    delay = min(60.0, (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue

            r.raise_for_status()
            resp = r.json()
            return resp["choices"][0]["message"]["content"]

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            # for 5xx/timeouts, backoff too
            delay = min(60.0, (2 ** attempt)) + random.uniform(0, 0.5)
            time.sleep(delay)

    raise RuntimeError(f"Databricks query failed after {max_retries} retries: {last_err}")



def prepare_dataset_for_openicl(
    train_inputs: List[str],
    train_targets: List[str],
    test_inputs: List[str],
    test_targets: List[str],
) -> DatasetReader:
    """
    Build DatasetReader for OpenICL. For Banking77, CLINC150, and HWU64, targets should be label names (strings).
    """
    train_ds = Dataset.from_dict({"input": train_inputs, "target": train_targets})
    test_ds = Dataset.from_dict({"input": test_inputs, "target": test_targets})
    dataset = DatasetDict({"train": train_ds, "test": test_ds})
    return DatasetReader(dataset, input_columns=["input"], output_column="target")


def run_openicl_retrievers(
    dataset_reader: DatasetReader,
    retrievers_to_run: List[str],
    ice_num: int,
    sentence_transformers_model: str,
    candidate_num: int,
    votek_k: int,
    mdl_ce_model: str,
    mdl_select_time: int,
    seed: int,
    cluster_ids: Optional[np.ndarray] = None,
    args: argparse.Namespace = None,
) -> Dict[str, Union[List[int], List[List[int]]]]:
    selections = {}
    for name in retrievers_to_run:
        # Aggressive memory clearing before each retriever
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logging.info(f"Running OpenICL retriever: {name}")
        try:
            if name == "dpp":
                retriever = DPPRetriever(
                    dataset_reader,
                    ice_num=ice_num,
                    candidate_num=candidate_num,
                    sentence_transformers_model_name=sentence_transformers_model,
                    seed=seed,
                )
            elif name == "mdl":
                # Extra aggressive clearing before MDL (most memory-intensive)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                logging.info(f"Using CE model for MDL: {mdl_ce_model}")
                retriever = MDLRetriever(
                    dataset_reader,
                    ice_num=ice_num,
                    candidate_num=candidate_num,
                    sentence_transformers_model_name=sentence_transformers_model,
                    ce_model_name=mdl_ce_model,
                    select_time=mdl_select_time,
                    seed=seed,
                )
            elif name == "votek":
                retriever = VotekRetriever(
                    dataset_reader,
                    ice_num=ice_num,
                    votek_k=votek_k,
                    sentence_transformers_model_name=sentence_transformers_model,
                )
            elif name == "votek_sgt":
                retriever = VotekSGTRetriever(
                    dataset_reader,
                    cluster_ids=cluster_ids,
                    ignore_label=args.sgt_ignore_label,
                    t_prior=int(args.sgt_t),
                    bin_size=args.sgt_bin_size,
                    smooth_count=False,
                    spectrum_model="powerlaw",
                    sgt_lambda=args.sgt_lambda,
                    ice_num=ice_num,
                    votek_k=args.votek_k,
                    sentence_transformers_model_name=args.sentence_transformers_model,
                    index_split="train",
                    test_split="test",
                    seed=seed,
                )
            elif name == "dpp_sgt":
                retriever = DPPSGTRetriever(
                    dataset_reader,
                    cluster_ids=cluster_ids,
                    sgt_lambda=args.sgt_lambda,
                    sgt_t=int(args.sgt_t),
                    sgt_bin_size=args.sgt_bin_size,
                    sgt_smooth_count=False,
                    sgt_offset=args.sgt_offset,
                    ignore_label=args.sgt_ignore_label,
                    sgt_score_type=args.sgt_score_type,
                    ice_num=ice_num,
                    candidate_num=args.candidate_num,
                    sentence_transformers_model_name=args.sentence_transformers_model,
                    seed=seed,
                    scale_factor=args.dpp_scale_factor,
                    index_split="train",
                    test_split="test",
                )
            elif name == "mdl_sgt":
                retriever = MDLSGTRetriever(
                    dataset_reader,
                    cluster_ids=cluster_ids,
                    sgt_lambda=args.sgt_lambda,
                    sgt_t=args.sgt_t,
                    sgt_bin_size=args.sgt_bin_size,
                    sgt_smooth_count=False,
                    sgt_offset=args.sgt_offset,
                    sgt_ignore_label=args.sgt_ignore_label,
                    ice_num=ice_num,
                    candidate_num=args.candidate_num,
                    sentence_transformers_model_name=args.sentence_transformers_model,
                    ce_model_name=args.mdl_ce_model,
                    select_time=args.mdl_select_time,
                    seed=seed,
                    index_split="train",
                    test_split="test",
                )
            else:
                raise ValueError(f"Unknown retriever: {name}")

            retrieved = retriever.retrieve()
            # Query-specific retrievers (dpp, mdl, dpp_sgt, mdl_sgt) return per-query selections
            # Corpus-level retrievers (votek, votek_sgt) return same selection for all queries
            query_specific_retrievers = {"dpp", "mdl", "dpp_sgt", "mdl_sgt"}
            if name in query_specific_retrievers:
                # Store all per-query selections
                selections[name] = retrieved  # List[List[int]] - each element is selection for one test example
                logging.info(f"{name} selected per-query: {len(retrieved)} queries, "
                           f"first query has {len(retrieved[0])} examples (first 10): {retrieved[0][:10]}")
            else:
                # Corpus-level: use first test example's selection for all
                selections[name] = retrieved[0]  # List[int] - same selection for all test examples
                logging.info(f"{name} selected {len(selections[name])} examples (first 10): {selections[name][:10]}")
            
            # Aggressive cleanup after retrieval
            if name in ["mdl", "mdl_sgt"]:
                # For MDL, try to free the metric model to save memory
                if hasattr(retriever, 'metric_model') and retriever.metric_model is not None:
                    del retriever.metric_model
                if hasattr(retriever, 'ce_model') and retriever.ce_model is not None:
                    del retriever.ce_model
                if hasattr(retriever, 'tokenizer') and retriever.tokenizer is not None:
                    del retriever.tokenizer
                logging.info("Freed MDL models from memory")
            
            # Delete retriever and all its attributes
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
            if hasattr(retriever, 'tokenizer') and retriever.tokenizer is not None:
                del retriever.tokenizer
            del retriever
            
            # Aggressive cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"OOM error during {name} retriever: {e}")
            logging.warning(f"Skipping {name} retriever due to OOM. Continuing with other retrievers...")
            # Clear cache and continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            # Don't add to selections, will be handled later
            continue
    
    return selections


def load_hf_chat_or_causal_model(
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool = True,
    token: Optional[str] = None,
) -> tuple:
    """
    Load HuggingFace tokenizer and model for generation.
    Handles different tokenizer requirements for various models.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
        token: HuggingFace token for gated models. If None, uses cached credentials or env vars.
    
    Returns:
        (tokenizer, model) tuple
    """
    logging.info(f"Loading HF model: {model_name} on {device} with dtype={dtype}")
    
    # Authenticate with HuggingFace if token is provided or available
    if login is not None:
        if token is not None:
            # Use provided token
            logging.info("Logging in to HuggingFace Hub with provided token")
            login(token=token)
        else:
            # Check for environment variables
            hf_token_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if hf_token_env:
                logging.info("Logging in to HuggingFace Hub with token from environment variable")
                login(token=hf_token_env)
            else:
                # Try to use cached credentials (from 'huggingface-cli login')
                try:
                    # login() without arguments uses cached credentials
                    login()
                    logging.info("Using HuggingFace Hub cached credentials")
                except Exception:
                    # No cached credentials - that's okay, transformers will try env vars or prompt
                    logging.debug("No cached HF credentials found. Will rely on transformers library authentication.")
    elif token is not None:
        logging.warning("huggingface_hub not installed. Token provided but cannot use login(). "
                       "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable instead.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    
    # Set padding token if missing (common for many models)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(tokenizer, "unk_token") and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Some models (like Gemma) may need special handling
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            logging.warning(f"Added <pad> token for {model_name}")
    
    # Determine torch dtype
    if dtype == "auto":
        # Use float16 for CUDA if available, else float32
        if device == "cuda" and torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:  # float32
        torch_dtype = torch.float32
    
    # Qwen2.5 models may have numerical instability with float16
    # Prefer bfloat16 if available (more stable than float16), else use float32
    if "qwen" in model_name.lower() and torch_dtype == torch.float16:
        if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logging.info(
                f"Qwen model detected: {model_name}. "
                f"Using bfloat16 instead of float16 to avoid numerical instability."
            )
            torch_dtype = torch.bfloat16
        else:
            logging.warning(
                f"Qwen model detected: {model_name}. "
                f"Using float32 instead of float16 to avoid numerical instability during generation."
            )
            torch_dtype = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device if device == "cuda" else None,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    
    # Move to device if not using device_map
    if device != "cuda" or not torch.cuda.is_available():
        model = model.to(device)
    
    model.eval()
    logging.info(f"Loaded HF model: {model_name}")
    
    return tokenizer, model


def hf_generate(
    prompt: str,
    tokenizer,
    model,
    system_text: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate text using HuggingFace model.
    
    Args:
        prompt: User prompt (demo block + query)
        tokenizer: HF tokenizer
        model: HF model
        system_text: System instruction
        max_input_tokens: Maximum input tokens (truncates from left, keeps rightmost tokens)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Generated text (only new tokens, excluding prompt)
    """
    # Combine system text with prompt
    full_prompt = f"{system_text}\n\n{prompt}"
    
    # Tokenize (without truncation first to check length)
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=False,
    )
    
    # Truncate from left (keep rightmost tokens) if needed
    input_ids = inputs["input_ids"][0]
    original_length = len(input_ids)
    if original_length > max_input_tokens:
        # Keep the rightmost max_input_tokens tokens
        input_ids = input_ids[-max_input_tokens:]
        # Reconstruct attention mask
        attention_mask = torch.ones_like(input_ids)
        inputs = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }
        logging.debug(f"Truncated input from {original_length} to {max_input_tokens} tokens (kept rightmost)")
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    # Disable cache for models that use HybridCache (like Gemma-2) to avoid torch._dynamo issues
    if model_name is None:
        model_name = getattr(model, 'name_or_path', '') if hasattr(model, 'name_or_path') else ''
        if not model_name:
            model_name = getattr(model.config, 'name_or_path', '') if hasattr(model.config, 'name_or_path') else ''
    model_name_lower = model_name.lower() if model_name else ''
    use_cache = not ("gemma" in model_name_lower or "gemma2" in model_name_lower)
    
    # For Qwen models with low temperature, use greedy decoding to avoid numerical issues
    # Temperature < 0.2 can cause numerical instability with float16
    use_greedy = False
   
    with torch.inference_mode():
        try:
            if use_greedy:
                # Use greedy decoding (temperature=0, no sampling)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
        except RuntimeError as e:
            if "probability tensor" in str(e) or "inf" in str(e).lower() or "nan" in str(e).lower():
                # Fallback to greedy decoding if sampling fails
                logging.warning(
                    f"Sampling failed for {model_name} (likely numerical instability). "
                    f"Falling back to greedy decoding."
                )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            else:
                raise
    
    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def hf_generate_batch(
    prompts: List[str],
    tokenizer,
    model,
    system_text: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Generate text for multiple prompts using HuggingFace model (batched).
    
    Args:
        prompts: List of user prompts (demo block + query)
        tokenizer: HF tokenizer
        model: HF model
        system_text: System instruction
        max_input_tokens: Maximum input tokens (truncates from left, keeps rightmost tokens)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        List of generated texts (only new tokens, excluding prompts)
    """
    if len(prompts) == 0:
        return []
    
    # Combine system text with each prompt
    full_prompts = [f"{system_text}\n\n{prompt}" for prompt in prompts]
    
    # Set padding_side to 'left' for decoder-only models (important for generation)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    try:
        # Tokenize each prompt individually, truncate, then pad to same length
        device = next(model.parameters()).device
        batch_size = len(prompts)
        processed_input_ids = []
        processed_attention_masks = []
        input_lengths = []
        
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Process each prompt individually
        for i, prompt in enumerate(full_prompts):
            # Tokenize this single prompt
            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
            )
            seq_input_ids = tokenized["input_ids"][0]  # Get first (and only) sequence
            
            original_length = len(seq_input_ids)
            if original_length > max_input_tokens:
                # Keep the rightmost max_input_tokens tokens
                seq_input_ids = seq_input_ids[-max_input_tokens:]
                logging.debug(f"Truncated input {i} from {original_length} to {max_input_tokens} tokens (kept rightmost)")
            
            # Create attention mask (all ones since no padding yet)
            seq_attention_mask = torch.ones(len(seq_input_ids), dtype=torch.long)
            
            processed_input_ids.append(seq_input_ids)
            processed_attention_masks.append(seq_attention_mask)
            input_lengths.append(len(seq_input_ids))
        
        # Find max length and pad all sequences to same length
        max_len = max(input_lengths) if input_lengths else 0
        
        # Pad all sequences to max_len (on the left for decoder-only models)
        padded_input_ids = []
        padded_attention_masks = []
        
        for i in range(batch_size):
            seq_input_ids = processed_input_ids[i]
            seq_attention_mask = processed_attention_masks[i]
            
            padding_length = max_len - len(seq_input_ids)
            if padding_length > 0:
                # Pad on the left (important for decoder-only models)
                padding_ids = torch.full((padding_length,), pad_token_id, dtype=seq_input_ids.dtype)
                padding_mask = torch.zeros(padding_length, dtype=seq_attention_mask.dtype)
                seq_input_ids = torch.cat([padding_ids, seq_input_ids])
                seq_attention_mask = torch.cat([padding_mask, seq_attention_mask])
            
            # Move to device
            seq_input_ids = seq_input_ids.to(device)
            seq_attention_mask = seq_attention_mask.to(device)
            
            padded_input_ids.append(seq_input_ids)
            padded_attention_masks.append(seq_attention_mask)
    finally:
        # Restore original padding_side
        tokenizer.padding_side = original_padding_side
    
    # Stack into batch tensors (all should be same size now)
    inputs = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
    }
    
    # Generate
    # Disable cache for models that use HybridCache (like Gemma-2) to avoid torch._dynamo issues
    if model_name is None:
        model_name = getattr(model, 'name_or_path', '') if hasattr(model, 'name_or_path') else ''
        if not model_name:
            model_name = getattr(model.config, 'name_or_path', '') if hasattr(model.config, 'name_or_path') else ''
    model_name_lower = model_name.lower() if model_name else ''
    use_cache = not ("gemma" in model_name_lower or "gemma2" in model_name_lower)
    
    # For Qwen models with low temperature, use greedy decoding to avoid numerical issues
    # Temperature < 0.2 can cause numerical instability with float16
    use_greedy = False
    
    with torch.inference_mode():
        try:
            if use_greedy:
                # Use greedy decoding (temperature=0, no sampling)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
        except RuntimeError as e:
            if "probability tensor" in str(e) or "inf" in str(e).lower() or "nan" in str(e).lower():
                # Fallback to greedy decoding if sampling fails
                logging.warning(
                    f"Sampling failed for {model_name} (likely numerical instability). "
                    f"Falling back to greedy decoding."
                )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            else:
                raise
    
    # Decode only the newly generated tokens for each item in batch
    # Note: Since we padded on the left, the actual input starts at (max_len - input_lengths[i])
    generated_texts = []
    for i in range(batch_size):
        input_length = input_lengths[i]
        # With left padding, the actual input starts at position (max_len - input_length)
        # So generated tokens start at max_len
        generated_tokens = outputs[i, max_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def run_icl_eval(
    train_texts: List[str],
    train_labels: List[Union[int, str]],
    test_texts: List[str],
    test_labels: List[Union[int, str]],
    label_names: List[str],
    generator_fn,
    dataset_type: str,
    backend_name: str = "unknown",
    system_text: str = "You are an assistant for intent classification.",
    batch_size: int = 16,
    batch_generator_fn=None,
) -> float:
    """
    Run ICL evaluation using a configurable generator function.

    Args:
        train_texts: List of training text examples
        train_labels: List of training targets (indices for classification, strings for BBEH)
        test_texts: List of test text examples
        test_labels: List of test targets (indices for classification, strings for BBEH)
        label_names: List of label names (classification datasets only)
        generator_fn: Function that takes a prompt string and returns generated text (for single prompts)
        dataset_type: Type of dataset ("banking77", "clinc150", "hwu64", or "bbeh")
        backend_name: Name of the backend (for logging)
        system_text: System message (used by some backends)
        batch_size: Batch size for processing (default: 1, no batching)
        batch_generator_fn: Function that takes a list of prompts and returns list of generated texts (for batching)
    """
    # Build demo block based on dataset type
    is_bbeh = dataset_type == "bbeh"
    if dataset_type == "banking77":
        demo_block = build_demo_block_banking77(train_texts, train_labels, label_names)
        build_prompt_fn = build_prompt_banking77
    elif dataset_type == "clinc150":
        demo_block = build_demo_block_clinc150(train_texts, train_labels, label_names)
        build_prompt_fn = build_prompt_clinc150
    elif dataset_type == "hwu64":
        demo_block = build_demo_block_hwu64(train_texts, train_labels, label_names)
        build_prompt_fn = build_prompt_hwu64
    elif dataset_type == "bbeh":
        demo_block = build_demo_block_bbh(train_texts, [str(x) for x in train_labels])
        build_prompt_fn = build_prompt_bbh
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    correct = 0
    total = len(test_texts)
    failed_matches = []
    
    use_batching = batch_size > 1 and batch_generator_fn is not None
    if use_batching:
        logging.info(f"Evaluating {total} test examples using {backend_name} backend with batch_size={batch_size}")
    else:
        logging.info(f"Evaluating {total} test examples using {backend_name} backend (no batching)")
    
    # Build all prompts upfront
    prompts = [build_prompt_fn(demo_block, query) for query in test_texts]
    
    # Process in batches or one by one
    if use_batching:
        # Process in batches
        for batch_start in tqdm(range(0, total, batch_size), desc=f"ICL Eval ({backend_name})"):
            batch_end = min(batch_start + batch_size, total)
            batch_prompts = prompts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            batch_true_labels = test_labels[batch_start:batch_end]
            
            try:
                # Generate responses for batch
                batch_responses = batch_generator_fn(batch_prompts)
                
                # Process each response in the batch
                for batch_idx, (response, true_target, i) in enumerate(zip(batch_responses, batch_true_labels, batch_indices)):
                    try:
                        # Clean and match - ensure response is a string
                        response_str = str(response) if not isinstance(response, str) else response
                        if is_bbeh:
                            cleaned = response_str.strip()
                            is_correct = evaluate_bbh_answer(cleaned, str(true_target))
                            pred_text = None
                        else:
                            cleaned = clean_generation(response_str)
                            pred_idx = best_label_match(cleaned, label_names)
                            is_correct = (pred_idx == true_target)
                            pred_text = label_names[pred_idx] if pred_idx is not None else None

                        if is_correct:
                            correct += 1
                        else:
                            # Log failed predictions for debugging (sample first 10)
                            if len(failed_matches) < 10:
                                failed_matches.append({
                                    "true_label": str(true_target) if is_bbeh else label_names[true_target],
                                    "pred_label": pred_text,
                                    "cleaned_output": cleaned,
                                    "raw_output": response[:100] if isinstance(response, str) else str(response)[:100],
                                })
                    except Exception as e:
                        logging.error(f"Error processing example {i} in batch: {e}")
                        # Count as incorrect
                        if len(failed_matches) < 10:
                            failed_matches.append({
                                "true_label": str(true_target) if is_bbeh else label_names[true_target],
                                "pred_label": None,
                                "error": str(e),
                            })
            except Exception as e:
                logging.error(f"Error generating batch starting at {batch_start}: {e}")
                # Count all in batch as incorrect
                for i, true_idx in zip(batch_indices, batch_true_labels):
                    if len(failed_matches) < 10:
                        failed_matches.append({
                            "true_label": label_names[true_idx],
                            "pred_label": None,
                            "error": str(e),
                        })
    else:
        # Process each test example one by one (original behavior)
        for i, (prompt, true_target) in enumerate(tqdm(zip(prompts, test_labels), total=total, desc=f"ICL Eval ({backend_name})")):
            # Generate response
            try:
                response = generator_fn(prompt)

                # Clean and match - ensure response is a string
                response_str = str(response) if not isinstance(response, str) else response
                if is_bbeh:
                    cleaned = response_str.strip()
                    is_correct = evaluate_bbh_answer(cleaned, str(true_target))
                    pred_text = None
                else:
                    cleaned = clean_generation(response_str)
                    pred_idx = best_label_match(cleaned, label_names)
                    is_correct = (pred_idx == true_target)
                    pred_text = label_names[pred_idx] if pred_idx is not None else None

                if is_correct:
                    correct += 1
                else:
                    # Log failed predictions for debugging (sample first 10)
                    if len(failed_matches) < 10:
                        failed_matches.append({
                            "true_label": str(true_target) if is_bbeh else label_names[true_target],
                            "pred_label": pred_text,
                            "cleaned_output": cleaned,
                            "raw_output": response[:100] if isinstance(response, str) else str(response)[:100],
                        })
            except Exception as e:
                logging.error(f"Error evaluating example {i}: {e}")
                # Count as incorrect
                if len(failed_matches) < 10:
                    failed_matches.append({
                        "true_label": str(true_target) if is_bbeh else label_names[true_target],
                        "pred_label": None,
                        "error": str(e),
                    })
    
    # Log summary of failures
    if failed_matches:
        logging.debug(f"Sample failed predictions (showing first {len(failed_matches)}):")
        for i, fail in enumerate(failed_matches[:5]):  # Show first 5
            logging.debug(f"  {i+1}. True: {fail.get('true_label', 'N/A')}, "
                         f"Pred: {fail.get('pred_label', 'N/A')}, "
                         f"Generated: '{fail.get('cleaned_output', 'N/A')}'")
    
    accuracy = correct / total
    logging.info(f"ICL Evaluation: {correct}/{total} correct ({accuracy:.4f})")
    if len(failed_matches) > 0:
        logging.info(f"Failed to match {total - correct} predictions. "
                    f"Check debug logs for examples.")
    
    return accuracy


def run_icl_eval_per_query(
    per_query_selections: List[List[int]],  # List of selections, one per test example
    train_inputs: List[str],
    train_targets: List[Union[int, str]],
    test_inputs: List[str],
    test_labels: List[Union[int, str]],
    label_names: List[str],
    generator_fn,
    dataset_type: str,
    backend_name: str = "unknown",
    system_text: str = "You are an assistant for intent classification.",
    batch_size: int = 16,
    batch_generator_fn=None,
) -> float:
    """
    Run ICL evaluation with per-query selections (each test example uses its own demo selection).

    Args:
        per_query_selections: List of selections, where per_query_selections[i] is the selection for test_inputs[i]
        train_inputs: List of training text examples
        train_targets: List of training targets (indices for classification, strings for BBEH)
        test_inputs: List of test text examples
        test_labels: List of test targets (indices for classification, strings for BBEH)
        label_names: List of label names (classification datasets only)
        generator_fn: Function that takes a prompt string and returns generated text (for single prompts)
        dataset_type: Type of dataset ("banking77", "clinc150", "hwu64", or "bbeh")
        backend_name: Name of the backend (for logging)
        system_text: System message (used by some backends)
        batch_size: Batch size for processing (default: 1, no batching)
        batch_generator_fn: Function that takes a list of prompts and returns list of generated texts (for batching)
    """
    if len(per_query_selections) != len(test_inputs):
        raise ValueError(f"Mismatch: per_query_selections has {len(per_query_selections)} items, "
                        f"but test_inputs has {len(test_inputs)} items")

    # Build prompt function based on dataset type
    is_bbeh = dataset_type == "bbeh"
    if dataset_type == "banking77":
        build_prompt_fn = build_prompt_banking77
    elif dataset_type == "clinc150":
        build_prompt_fn = build_prompt_clinc150
    elif dataset_type == "hwu64":
        build_prompt_fn = build_prompt_hwu64
    elif dataset_type == "bbeh":
        build_prompt_fn = build_prompt_bbh
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    correct = 0
    total = len(test_inputs)
    failed_matches = []
    
    use_batching = batch_size > 1 and batch_generator_fn is not None
    if use_batching:
        n_batches = (total + batch_size - 1) // batch_size
        logging.info(f"Evaluating {total} test examples with per-query selections using {backend_name} backend with batch_size={batch_size} ({n_batches} batches)")
    else:
        logging.info(f"Evaluating {total} test examples with per-query selections using {backend_name} backend (no batching, batch_size={batch_size}, batch_generator_fn={'None' if batch_generator_fn is None else 'provided'})")
    
    # Helper function to build a single prompt
    def build_single_prompt(query_idx: int) -> str:
        """Build a single prompt for a query."""
        query = test_inputs[query_idx]
        sel_indices = per_query_selections[query_idx]
        sel_inputs = [train_inputs[idx] for idx in sel_indices]
        sel_targets = [train_targets[idx] for idx in sel_indices]

        if dataset_type == "banking77":
            demo_block = build_demo_block_banking77(sel_inputs, sel_targets, label_names)
        elif dataset_type == "clinc150":
            demo_block = build_demo_block_clinc150(sel_inputs, sel_targets, label_names)
        elif dataset_type == "hwu64":
            demo_block = build_demo_block_hwu64(sel_inputs, sel_targets, label_names)
        elif dataset_type == "bbeh":
            demo_block = build_demo_block_bbh(sel_inputs, [str(x) for x in sel_targets])
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        return build_prompt_fn(demo_block, query)
    
    # Process in batches or one by one
    if use_batching:
        # Optimized: Build prompts lazily in parallel during batching
        # This reduces memory usage and allows parallelization
        prompt_build_start = time.time()
        
        # Pre-build prompts in parallel for better batching efficiency
        # Use ThreadPoolExecutor for I/O-bound string operations
        max_workers = min(8, total)  # Limit parallelism to avoid overhead
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            prompts = list(tqdm(
                executor.map(build_single_prompt, range(total)),
                total=total,
                desc="Building prompts (parallel)"
            ))
        
        prompt_build_time = time.time() - prompt_build_start
        logging.info(f"Built {total} prompts in {prompt_build_time:.2f}s ({prompt_build_time/total*1000:.1f}ms per prompt)")
        
        # Process in batches
        n_batches = (total + batch_size - 1) // batch_size
        generation_start = time.time()
        for batch_start in tqdm(range(0, total, batch_size), total=n_batches, desc=f"ICL Eval Per-Query ({backend_name}, batch_size={batch_size})"):
            batch_end = min(batch_start + batch_size, total)
            batch_prompts = prompts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            batch_true_labels = test_labels[batch_start:batch_end]
            
            try:
                # Generate responses for batch
                batch_responses = batch_generator_fn(batch_prompts)
                
                # Process each response in the batch
                for batch_idx, (response, true_target, i) in enumerate(zip(batch_responses, batch_true_labels, batch_indices)):
                    try:
                        # Clean and match - ensure response is a string
                        response_str = str(response) if not isinstance(response, str) else response
                        if is_bbeh:
                            cleaned = response_str.strip()
                            is_correct = evaluate_bbh_answer(cleaned, str(true_target))
                            pred_text = None
                        else:
                            cleaned = clean_generation(response_str)
                            pred_idx = best_label_match(cleaned, label_names)
                            is_correct = (pred_idx == true_target)
                            pred_text = label_names[pred_idx] if pred_idx is not None else None

                        if is_correct:
                            correct += 1
                        else:
                            # Log failed predictions for debugging (sample first 10)
                            if len(failed_matches) < 10:
                                failed_matches.append({
                                    "true_label": str(true_target) if is_bbeh else label_names[true_target],
                                    "pred_label": pred_text,
                                    "cleaned_output": cleaned,
                                    "raw_output": response[:100] if isinstance(response, str) else str(response)[:100],
                                })
                    except Exception as e:
                        logging.error(f"Error processing example {i} in batch: {e}")
                        # Count as incorrect
                        if len(failed_matches) < 10:
                            failed_matches.append({
                                "true_label": str(true_target) if is_bbeh else label_names[true_target],
                                "pred_label": None,
                                "error": str(e),
                            })
            except Exception as e:
                logging.error(f"Error generating batch starting at {batch_start}: {e}")
                # Count all in batch as incorrect
                for i, true_idx in zip(batch_indices, batch_true_labels):
                    if len(failed_matches) < 10:
                        failed_matches.append({
                            "true_label": label_names[true_idx],
                            "pred_label": None,
                            "error": str(e),
                        })
        
        generation_time = time.time() - generation_start
        logging.info(f"Generation completed in {generation_time:.2f}s ({generation_time/total*1000:.1f}ms per query, {generation_time/n_batches:.2f}s per batch)")
    else:
        # Process each test example one by one (original behavior)
        # Build prompts on-demand to save memory
        for i, (query, sel_indices, true_idx) in enumerate(tqdm(
            zip(test_inputs, per_query_selections, test_labels), 
            total=total, 
            desc=f"ICL Eval Per-Query ({backend_name})"
        )):
            # Build prompt on-demand
            prompt = build_single_prompt(i)
            # Generate response
            try:
                response = generator_fn(prompt)
                
                # Clean and match - ensure response is a string
                response_str = str(response) if not isinstance(response, str) else response
                cleaned = clean_generation(response_str)
                pred_idx = best_label_match(cleaned, label_names)
                
                is_correct = (pred_idx == true_idx)
                
                if is_correct:
                    correct += 1
                else:
                    # Log failed predictions for debugging (sample first 10)
                    if len(failed_matches) < 10:
                        failed_matches.append({
                            "true_label": label_names[true_idx],
                            "pred_label": label_names[pred_idx] if pred_idx is not None else None,
                            "cleaned_output": cleaned,
                            "raw_output": response[:100] if isinstance(response, str) else str(response)[:100],
                        })
            except Exception as e:
                logging.error(f"Error evaluating example {i}: {e}")
                # Count as incorrect
                if len(failed_matches) < 10:
                    failed_matches.append({
                        "true_label": label_names[true_idx],
                        "pred_label": None,
                        "error": str(e),
                    })
    
    # Log summary of failures
    if failed_matches:
        logging.debug(f"Sample failed predictions (showing first {len(failed_matches)}):")
        for i, fail in enumerate(failed_matches[:5]):  # Show first 5
            logging.debug(f"  {i+1}. True: {fail.get('true_label', 'N/A')}, "
                         f"Pred: {fail.get('pred_label', 'N/A')}, "
                         f"Generated: '{fail.get('cleaned_output', 'N/A')}'")
    
    accuracy = correct / total
    logging.info(f"ICL Evaluation (per-query): {correct}/{total} correct ({accuracy:.4f})")
    if len(failed_matches) > 0:
        logging.info(f"Failed to match {total - correct} predictions. "
                    f"Check debug logs for examples.")
    
    return accuracy


# Keep for backward compatibility (if needed elsewhere)
def run_icl_eval_databricks(
    endpoint_name: str,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    label_names: List[str],
    max_tokens: int = 64,
    system_text: str = "You are an assistant for intent classification.",
) -> float:
    """Legacy wrapper for Databricks evaluation."""
    generator_fn = lambda prompt: query_dbx_chat(
        endpoint=endpoint_name,
        user_text=prompt,
        system_text=system_text,
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return run_icl_eval(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=label_names,
        generator_fn=generator_fn,
        backend_name=f"Databricks ({endpoint_name})",
        system_text=system_text,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare clustering+selection vs random and OpenICL retrievers. "
                    "Supports both Databricks endpoints and local HuggingFace models for evaluation."
    )

    # Core
    parser.add_argument("--dataset_type", type=str, default="banking77", choices=["banking77", "clinc150", "hwu64", "bbeh"])
    parser.add_argument("--bbeh_task", type=str, default="bbeh_boolean_expressions",
                        help="BBEH task name (only used if dataset_type=bbeh)")
    parser.add_argument("--bbeh_data_dir", type=str, default=None,
                        help="Directory containing BBEH data (optional, uses default if not provided)")
    parser.add_argument("--endpoint_name", type=str, required=True,
                        help="For databricks: endpoint name. For hf: model name (if --eval_model_name not provided)")
    parser.add_argument("--embedding_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help="Causal LM for computing embeddings (can be smaller than endpoint model)")
    
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer to use for embeddings. -1 means last layer, 0 means first layer, etc.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "first", "last"],
                        help="Pooling method to use for embeddings. 'mean' means average pooling, 'first' means first token, 'last' means last non-padding token.")
    parser.add_argument("--l2_normalize", action="store_true", default=False,
                        help="L2 normalize LLM embeddings (default: off)")
    
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=16,
                        help="Max tokens to generate (used for both backends, default: 16 for intent classification)")
    
    # Evaluation backend
    parser.add_argument("--eval_backend", type=str, default="hf", choices=["databricks", "hf"],
                        help="Evaluation backend: 'databricks' for Databricks endpoints, 'hf' for local HuggingFace models (default: hf)")
    parser.add_argument("--eval_model_name", type=str, default=None,
                        help="HF model name for evaluation (if hf backend). Defaults to --endpoint_name if not provided")
    parser.add_argument("--hf_device", type=str, default=None,
                        help="Device for HF model (default: 'cuda' if available, else 'cpu')")
    parser.add_argument("--hf_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                        help="Data type for HF model (default: auto)")
    parser.add_argument("--hf_max_input_tokens", type=int, default=16384,
                        help="Max input tokens for HF model (truncates from left, keeps rightmost tokens)")
    parser.add_argument("--hf_temperature", type=float, default=0.1,
                        help="Temperature for HF generation")
    parser.add_argument("--hf_top_p", type=float, default=1.0,
                        help="Top-p for HF generation")
    parser.add_argument("--hf_max_new_tokens", type=int, default=None,
                        help="Max new tokens for HF generation (defaults to --max_tokens)")
    parser.add_argument("--icl_batch_size", type=int, default=4,
                        help="Batch size for ICL evaluation (default: 1, no batching). Use larger values (e.g., 4-16) for faster evaluation with HF backend")
    parser.add_argument("--hf_trust_remote_code", action="store_true", default=True,
                        help="Trust remote code when loading HF model (default: True)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models. Can also set HF_TOKEN or HUGGINGFACE_HUB_TOKEN env var, or use 'huggingface-cli login'")

    # Which systems to evaluate (orthogonal to clustering/selection)
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["random", "dpp", "mdl", "votek", "dpp_sgt", "mdl_sgt", "votek_sgt"],
        choices=["none", "random", "dpp", "mdl", "votek", "dpp_sgt", "mdl_sgt", "votek_sgt"],
        help="Baselines to run. Use 'none' to disable all baselines.",
    )
    parser.add_argument("--skip_mdl", action="store_true",
                        help="Skip MDL (memory intensive). Equivalent to removing 'mdl' from --baselines.")
    parser.add_argument("--sentence_transformers_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--candidate_num", type=int, default=50)
    parser.add_argument("--votek_k", type=int, default=3)
    parser.add_argument("--mdl_ce_model", type=str, default="gpt2")
    parser.add_argument("--mdl_select_time", type=int, default=5)
    # NEW: clustering method (how to define “knowledge bins / clusters”)
    # - dbscan: cluster_ids from run_dbscan_thresholded
    # - dict_argmax: dictionary learning then cluster_ids = argmax(|R|)
    parser.add_argument(
        "--clustering",
        type=str,
        default="dict_argmax",
        choices=["dbscan", "dict_argmax", "dict_dbscan"],
        help="How to produce cluster_ids for SGT-style selection.",
    )

    # dbscan knobs
    parser.add_argument("--dbscan_k", type=int, default=10)
    parser.add_argument("--robust_sgt_top_m", type=int, default=20)
    parser.add_argument("--dbscan_q", type=float, default=0.05)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)

    # dict learning knobs (used when clustering=dict_dbscan)
    parser.add_argument("--dict_n_components", type=int, default=64)
    parser.add_argument("--dict_alpha", type=float, default=1.0)
    parser.add_argument("--dict_top_k", type=int, default=4)
    parser.add_argument("--dict_tau", type=float, default=1e-3)

    parser.add_argument("--dict_standardize", action="store_true", default=False,
                        help="Standardize embeddings before dictionary learning (default: off)")
    parser.add_argument("--no_dict_standardize", dest="dict_standardize", action="store_false",
                        help="Disable standardization before dictionary learning")

    parser.add_argument("--dict_use_minibatch", action="store_true", default=True)
    parser.add_argument("--no_dict_use_minibatch", dest="dict_use_minibatch", action="store_false")
    parser.add_argument("--dict_batch_size", type=int, default=512)
    parser.add_argument("--dict_pca_dim", type=int, default=512,
                        help="PCA dim before dict learning (<=0 disables PCA)")
    parser.add_argument("--dict_transform_algorithm", type=str, default="ridge", choices=["omp", "lasso_cd", "ridge"],
                        help="Transform algorithm: 'omp' (fixed sparsity), 'lasso_cd' (L1, sparse), 'ridge' (L2, less sparse)")
    parser.add_argument("--dict_regularization_type", type=str, default="l2", choices=["l1", "l2"],
                        help="Regularization type for dictionary learning: 'l1' (sparse) or 'l2' (less sparse). Note: only affects transform when using 'ridge' algorithm.")
    parser.add_argument("--dict_max_iter", type=int, default=100,
                        help="Max iterations for dictionary learning (default: 100, reduced from 300 for speed)")
    
    parser.add_argument("--sgt_lambda", type=float, default=0.1,
                        help="Weight for SGT prior in dpp_sgt, mdl_sgt, and votek_sgt retrievers (default: 0.1)")
    parser.add_argument("--sgt_ignore_label", type=int, default=-1)
    parser.add_argument("--dpp_scale_factor", type=float, default=0.1)
    parser.add_argument("--sgt_score_type", type=str, default="seen+unseen",
                        choices=["seen+unseen", "seen_only", "unseen_only", "unseen_ratio"],
                        help="SGT score type for ablation study (default: seen+unseen). "
                             "Options: seen+unseen (full SGT), seen_only (distinct clusters), "
                             "unseen_only (estimated unseen), unseen_ratio (unseen/seen)")

    

    # NEW: selection method (how to pick demos)
    # - sgt_standard / sgt_fast: uses cluster_ids + greedy SGT
    # - common_rare: dictionary active sets + freq (requires dict clustering)
    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=["sgt_standard", "sgt_fast"],
        choices=["sgt_standard", "sgt_fast", "common_rare", "robust_sgt"],
        help="Selection algorithm(s) to run.",
    )

    # SGT knobs
    parser.add_argument("--sgt_seed_size", type=int, default=5)
    parser.add_argument("--sgt_t", type=int, default=3)
    parser.add_argument("--sgt_bin_size", type=int, default=20)
    parser.add_argument("--sgt_smooth_count", action="store_true", default=False)
    parser.add_argument("--sgt_offset", type=float, default=1.0)

    # common+rare knobs (requires dict clustering)
    parser.add_argument("--frac_rare", type=float, default=0.4)
    parser.add_argument("--rare_quantile", type=float, default=0.3)
    parser.add_argument("--common_quantile", type=float, default=0.7)
    parser.add_argument("--lambda_rep", type=float, default=0.15)

    args = parser.parse_args()

    # normalize baselines
    if "none" in args.baselines:
        args.baselines = []
    if args.skip_mdl and "mdl" in args.baselines:
        args.baselines = [b for b in args.baselines if b != "mdl"]

    return args



def main():
    args = parse_args()

    dataset_type = args.dataset_type
    seed = args.seed
    endpoint_name = args.endpoint_name
    embedding_model_name = args.embedding_model_name
    budget = args.budget
    test_size = args.test_size
    n_runs = args.n_runs
    eval_backend = args.eval_backend

    # Validate backend and set up evaluation model
    if eval_backend == "databricks":
        # Validate Databricks environment variables
        if "DATABRICKS_HOST" not in os.environ:
            raise ValueError("DATABRICKS_HOST environment variable not set (required for databricks backend)")
        if "DATABRICKS_TOKEN" not in os.environ:
            raise ValueError("DATABRICKS_TOKEN environment variable not set (required for databricks backend)")
        eval_model_name = endpoint_name
        logging.info(f"Using Databricks backend with endpoint: {endpoint_name}")
    elif eval_backend == "hf":
        # Determine HF model name
        if args.eval_model_name:
            eval_model_name = args.eval_model_name
        else:
            # Try to infer from endpoint_name
            eval_model_name = endpoint_name
            logging.info(f"Using --endpoint_name as HF model name: {eval_model_name}")
        
        # Validate HF model name looks reasonable
        if not eval_model_name or len(eval_model_name) < 3:
            raise ValueError(
                f"Invalid HF model name: {eval_model_name}. "
                f"Please provide --eval_model_name or use a valid model name in --endpoint_name"
            )
        
        # Determine device
        hf_device = args.hf_device
        if hf_device is None:
            hf_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set max_new_tokens
        hf_max_new_tokens = args.hf_max_new_tokens if args.hf_max_new_tokens is not None else args.max_tokens
        
        logging.info(
            f"Using HuggingFace backend with model: {eval_model_name}, "
            f"device: {hf_device}, dtype: {args.hf_dtype}, "
            f"max_input_tokens: {args.hf_max_input_tokens}, "
            f"max_new_tokens: {hf_max_new_tokens}"
        )
    else:
        raise ValueError(f"Unknown eval_backend: {eval_backend}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:4] # Generate a unique run ID
    endpoint_short = endpoint_name.replace("/", "_").replace("-", "_")
    # Include sgt_score_type in output dir name if using dpp_sgt (for ablation study)
    score_type_suffix = f"_{args.sgt_score_type}" if "dpp_sgt" in args.baselines else ""
    output_dir = Path("outputs") / f"{dataset_type}_{endpoint_short}_{args.clustering}{score_type_suffix}_{timestamp}_{run_id}"
    setup_logging(output_dir)
    
    logging.info(
        f"Dataset: {dataset_type}, "
        f"Eval Backend: {eval_backend}, "
        f"Eval Model/Endpoint: {eval_model_name}, "
        f"Embedding Model: {embedding_model_name}, "
        f"L2 Normalize: {args.l2_normalize}, "
        f"Budget: {budget}, Test size: {test_size}, N runs: {n_runs}, "
        f"Baselines: {args.baselines}, "
        f"Clustering: {args.clustering}, "
        f"Selection: {args.selection}, "
        f"SGT Score Type: {args.sgt_score_type}"
    )

    # Load data - use actual train/test splits to avoid data leakage
    if dataset_type == "banking77":
        train_texts, train_labels = load_banking77("train")
        test_texts, test_labels = load_banking77("test")
        label_names = load_banking77_label_names()
        dataset_cache_name = "banking77_train_full"
    elif dataset_type == "clinc150":
        train_texts, train_labels = load_clinc150("train")
        test_texts, test_labels = load_clinc150("test")
        label_names = load_clinc150_label_names()
        dataset_cache_name = "clinc150_train_full"
    elif dataset_type == "hwu64":
        train_texts, train_labels = load_hwu64("train")
        test_texts, test_labels = load_hwu64("test")
        label_names = load_hwu64_label_names()
        dataset_cache_name = "hwu64_train_full"
    elif dataset_type == "bbeh":
        bbeh_data_dir = Path(args.bbeh_data_dir) if args.bbeh_data_dir else None
        all_inputs, all_targets = load_bbeh_task(args.bbeh_task, data_dir=bbeh_data_dir)
        n_examples = len(all_inputs)
        if n_examples < 2:
            raise ValueError(f"BBEH task '{args.bbeh_task}' needs at least 2 examples, got {n_examples}")

        # Deterministic split: first 2/3 for train, last 1/3 for test
        split_idx = int(2 * n_examples / 3)
        train_texts = all_inputs[:split_idx]
        train_labels = all_targets[:split_idx]
        test_texts = all_inputs[split_idx:]
        test_labels = all_targets[split_idx:]
        label_names = []  # Not used for BBEH, but keep for compatibility
        dataset_cache_name = f"{args.bbeh_task}_train_full"
        logging.info(
            f"Loaded BBEH task '{args.bbeh_task}' with deterministic split: "
            f"{len(train_texts)} train, {len(test_texts)} test"
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Use train data for the pool (for selection)
    train_inputs = train_texts
    train_targets = train_labels
    train_pool_size = len(train_texts)
    
    # Use test data for evaluation
    test_inputs = test_texts
    test_targets = test_labels
    actual_test_size = len(test_texts)
    
    # If test_size was specified, optionally subsample test set (for faster evaluation)
    if test_size > 0 and test_size < actual_test_size:
        logging.info(f"Subsampling test set from {actual_test_size} to {test_size} examples")
        test_indices = np.random.RandomState(seed).choice(actual_test_size, size=test_size, replace=False)
        test_inputs = [test_texts[i] for i in test_indices]
        test_targets = [test_labels[i] for i in test_indices]
        actual_test_size = test_size
    elif test_size > 0 and test_size >= actual_test_size:
        logging.info(f"Requested test_size ({test_size}) >= actual test set size ({actual_test_size}). Using full test set.")
        actual_test_size = len(test_texts)
    
    texts_for_embeddings = train_texts
    
    # Ensure we have enough data for the budget
    min_train_pool = budget if dataset_type == "bbeh" else (budget + 1)
    if train_pool_size < min_train_pool:
        raise ValueError(
            f"Train pool ({train_pool_size}) is smaller than minimum required ({min_train_pool}). "
            f"Reduce budget or use a dataset with more training examples."
        )
    
    logging.info(f"Train pool: {train_pool_size} examples, Test set: {actual_test_size} examples")

    # Compute or load cached embeddings using shared helper (only for train data)
    cache_root = Path("outputs") / "LLM_embeddings"
    train_embeddings = get_or_compute_embeddings(
        texts=texts_for_embeddings,
        model_name=embedding_model_name,
        cache_root=cache_root,
        dataset_name=dataset_cache_name,
        layer=args.layer,            # use last layer to match previous behavior
        batch_size=16,       # keep previous batch size for efficiency
        max_length=256,
        pooling=args.pooling,
        l2_normalize=args.l2_normalize,
    )
    logging.info(f"Train embeddings shape: {train_embeddings.shape}")
    
    # Load evaluation model if using HF backend (load once, reuse across runs)
    hf_tokenizer = None
    hf_model = None
    if dataset_type == "bbeh":
        system_text = "You are a helpful assistant that answers reasoning questions concisely."
    else:
        system_text = "You are an assistant for intent classification."
    
    if eval_backend == "hf":
        logging.info("Loading HuggingFace model for evaluation...")
        hf_tokenizer, hf_model = load_hf_chat_or_causal_model(
            model_name=eval_model_name,
            device=hf_device,
            dtype=args.hf_dtype,
            trust_remote_code=args.hf_trust_remote_code,
            token=args.hf_token,
        )
        # Free embedding model memory if it exists (to save GPU memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logging.info("HF model loaded and ready for evaluation")
    
    # Store results across runs
    # Keys will be filled dynamically, e.g.:
    #  - "baseline_random", "baseline_openicl_dpp", ...
    all_results: Dict[str, List[Optional[float]]] = {}
    cluster_counts_per_run = []
    # Store selected indices for each run and method (for analysis)
    # For corpus-level methods: List[List[int]] (runs × indices)
    # For per-query methods: List[List[List[int]]] (runs × queries × indices)
    all_selected_indices: Dict[str, Union[List[List[int]], List[List[List[int]]]]] = {}
    # Note: pool_indices is now the same for all runs (full training set)
    pool_indices = np.arange(train_pool_size).tolist()

    # Run multiple times with different random seeds (for clustering/selection randomness)
    # but using the same train/test split to avoid data leakage
    for run_idx in range(n_runs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Run {run_idx + 1}/{n_runs}")
        logging.info(f"{'='*60}")
        
        # Use different seed for each run (affects clustering/selection randomness)
        run_seed = seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        
        logging.info(f"Run {run_idx + 1}: Train pool: {len(train_inputs)}, Test: {len(test_inputs)}")
        
        # Final validation
        if len(train_inputs) == 0:
            raise ValueError(f"No training examples available! Train pool size: {train_pool_size}")
        if len(train_inputs) < budget:
            raise ValueError(f"Train pool ({len(train_inputs)}) is smaller than budget ({budget}). "
                            f"Reduce budget.")

        # -------------------------
        # 0) Helpers
        # -------------------------
        def _validate_and_pad(indices: List[int], pool_size: int, k: int, rng: random.Random) -> List[int]:
            idx = [int(i) for i in indices if 0 <= int(i) < pool_size]
            # unique while preserving order
            seen = set()
            idx2: List[int] = []
            for i in idx:
                if i not in seen:
                    seen.add(i)
                    idx2.append(i)
            if len(idx2) >= k:
                return idx2[:k]
            # pad with random unseen indices
            remaining = [i for i in range(pool_size) if i not in seen]
            need = k - len(idx2)
            if need > 0 and remaining:
                idx2.extend(rng.sample(remaining, min(need, len(remaining))))
            return idx2
        
        # Set up generator function for this run
        if eval_backend == "databricks":
            generator_fn = lambda prompt: query_dbx_chat(
                endpoint=endpoint_name,
                user_text=prompt,
                system_text=system_text,
                temperature=0.1,
                max_tokens=args.max_tokens,
            )
            batch_generator_fn = None  # Databricks doesn't support batching
            backend_display_name = f"Databricks ({endpoint_name})"
        elif eval_backend == "hf":
            # Use the pre-loaded model
            hf_max_new_tokens = args.hf_max_new_tokens if args.hf_max_new_tokens is not None else args.max_tokens
            generator_fn = lambda prompt: hf_generate(
                prompt=prompt,
                tokenizer=hf_tokenizer,
                model=hf_model,
                system_text=system_text,
                max_input_tokens=args.hf_max_input_tokens,
                max_new_tokens=hf_max_new_tokens,
                temperature=args.hf_temperature,
                top_p=args.hf_top_p,
                model_name=eval_model_name,
            )
            # Set up batch generator if batch_size > 1
            if args.icl_batch_size > 1:
                batch_generator_fn = lambda prompts: hf_generate_batch(
                    prompts=prompts,
                    tokenizer=hf_tokenizer,
                    model=hf_model,
                    system_text=system_text,
                    max_input_tokens=args.hf_max_input_tokens,
                    max_new_tokens=hf_max_new_tokens,
                    temperature=args.hf_temperature,
                    top_p=args.hf_top_p,
                    model_name=eval_model_name,
                )
            else:
                batch_generator_fn = None
            backend_display_name = f"HF ({eval_model_name})"
        else:
            raise ValueError(f"Unknown eval_backend: {eval_backend}")
        
        def eval_selection(sel_indices: Union[List[int], List[List[int]]], label: str) -> float:
            """
            Evaluate a selection. Handles both corpus-level (List[int]) and per-query (List[List[int]]) selections.
            """
            if len(sel_indices) == 0:
                raise ValueError(f"Empty selection for {label}")
            
            # Check if this is per-query selection (List[List[int]]) or corpus-level (List[int])
            if isinstance(sel_indices[0], list):
                # Per-query selection: List[List[int]]
                per_query_selections = sel_indices
                if len(per_query_selections) != len(test_inputs):
                    raise ValueError(f"Mismatch for {label}: per_query_selections has {len(per_query_selections)} items, "
                                   f"but test_inputs has {len(test_inputs)} items")
                
                # Validate all selections
                max_idx = len(train_inputs) - 1
                for query_idx, query_sel in enumerate(per_query_selections):
                    invalid = [i for i in query_sel if i < 0 or i > max_idx]
                    if invalid:
                        raise ValueError(f"Invalid indices for {label} query {query_idx}: {invalid} (valid range: 0-{max_idx})")
                
                # Use per-query evaluation
                acc = run_icl_eval_per_query(
                    per_query_selections=per_query_selections,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    test_inputs=test_inputs,
                    test_labels=test_targets,
                    label_names=label_names,
                    generator_fn=generator_fn,
                    dataset_type=dataset_type,
                    backend_name=backend_display_name,
                    system_text=system_text,
                    batch_size=args.icl_batch_size,
                    batch_generator_fn=batch_generator_fn,
                )
            else:
                # Corpus-level selection: List[int]
                max_idx = len(train_inputs) - 1
                invalid = [i for i in sel_indices if i < 0 or i > max_idx]
                if invalid:
                    raise ValueError(f"Invalid indices for {label}: {invalid} (valid range: 0-{max_idx})")
            
                sel_inputs = [train_inputs[i] for i in sel_indices]
                sel_targets = [train_targets[i] for i in sel_indices]
            
                acc = run_icl_eval(
                    train_texts=sel_inputs,
                    train_labels=sel_targets,
                    test_texts=test_inputs,
                    test_labels=test_targets,
                    label_names=label_names,
                    generator_fn=generator_fn,
                    dataset_type=dataset_type,
                    backend_name=backend_display_name,
                    system_text=system_text,
                    batch_size=args.icl_batch_size,
                    batch_generator_fn=batch_generator_fn,
                )
            logging.info(f"Run {run_idx + 1} - {label}: {acc:.4f}")
            return acc
        
        run_rng = random.Random(run_seed)
        
        # -------------------------
        # 1) Clustering → produce cluster_ids (+ dict artifacts if needed)
        # -------------------------
        cluster_ids = None
        n_clusters = None
        
        # dict artifacts (only present for dict clustering)
        dict_active_sets = None
        dict_atom_freq = None
        dict_R = None
        
        if args.clustering == "dbscan":
            logging.info(f"Run {run_idx + 1}: Clustering = DBSCAN")
            cluster_ids, eps = run_dbscan_thresholded(
                train_embeddings,
                k=args.dbscan_k,
                q=args.dbscan_q,
                min_samples=args.dbscan_min_samples,
                verbose=True,
            )
        
            # map noise -1 to singleton clusters (each noise point gets its own cluster ID)
            n_noise = int(np.sum(cluster_ids == -1))
            if n_noise > 0:
                max_cluster_id = int(np.max(cluster_ids[cluster_ids >= 0])) if np.any(cluster_ids >= 0) else -1
                cluster_ids = cluster_ids.copy()
                # Assign each noise point a unique cluster ID (singleton clusters)
                noise_indices = np.where(cluster_ids == -1)[0]
                for i, noise_idx in enumerate(noise_indices):
                    cluster_ids[noise_idx] = max_cluster_id + 1 + i
                logging.info(f"Assigned {n_noise} noise points to {n_noise} singleton clusters (IDs: {max_cluster_id + 1} to {max_cluster_id + n_noise})")
        
            if len(cluster_ids) != len(train_inputs):
                raise ValueError(f"Mismatch: cluster_ids={len(cluster_ids)} vs train_inputs={len(train_inputs)}")
        
            n_clusters = int(len(np.unique(cluster_ids)))
            logging.info(f"Run {run_idx + 1}: DBSCAN produced {n_clusters} clusters")
            plot_cluster_distribution(cluster_ids, output_dir, run_idx)
        
            # record cluster count
            if len(cluster_counts_per_run) <= run_idx:
                cluster_counts_per_run.append(n_clusters)
            else:
                cluster_counts_per_run[run_idx] = n_clusters
        
        elif args.clustering == "dict_argmax":
            logging.info(f"Run {run_idx + 1}: Clustering = Dictionary(argmax |R|)")
        
            dk_cfg = DictKnowledgeConfig(
                n_components=args.dict_n_components,
                alpha=args.dict_alpha,
                regularization_type=args.dict_regularization_type,
                top_k=args.dict_top_k,
                tau=args.dict_tau,
                standardize=args.dict_standardize,
                random_state=run_seed,
                use_minibatch=args.dict_use_minibatch,
                batch_size=args.dict_batch_size,
                pca_dim=(None if args.dict_pca_dim <= 0 else args.dict_pca_dim),
                transform_algorithm=args.dict_transform_algorithm,
                max_iter=args.dict_max_iter,
            )
            km = fit_dictionary_knowledge(train_embeddings, dk_cfg)
        
            dict_R = km.transform_codes(train_embeddings)
            dict_active_sets = km.active_atoms(dict_R)
            dict_atom_freq = atom_frequencies(dict_active_sets, n_atoms=dk_cfg.n_components)
        
            absR = np.abs(dict_R)
            cluster_ids = absR.argmax(axis=1).astype(np.int32)
        
            n_clusters = int(len(np.unique(cluster_ids)))
            logging.info(f"Run {run_idx + 1}: Dict clustering produced {n_clusters} argmax-atoms")
            # Plot and save cluster distribution for dictionary-based clustering as well
            plot_cluster_distribution(cluster_ids, output_dir, run_idx)
            if len(cluster_counts_per_run) <= run_idx:
                cluster_counts_per_run.append(n_clusters)
            else:
                cluster_counts_per_run[run_idx] = n_clusters
        
        elif args.clustering == "dict_dbscan":
            logging.info(f"Run {run_idx + 1}: Clustering = Dictionary Learning + DBSCAN")
            logging.info(f"Run {run_idx + 1}: Robust SGT top_m = {args.robust_sgt_top_m}")
        
            dk_cfg = DictKnowledgeConfig(
                n_components=args.dict_n_components,
                alpha=args.dict_alpha,
                regularization_type=args.dict_regularization_type,
                top_k=args.dict_top_k,
                tau=args.dict_tau,
                standardize=args.dict_standardize,
                random_state=run_seed,
                use_minibatch=args.dict_use_minibatch,
                batch_size=args.dict_batch_size,
                pca_dim=(None if args.dict_pca_dim <= 0 else args.dict_pca_dim),
                transform_algorithm=args.dict_transform_algorithm,
                max_iter=args.dict_max_iter,
            )
            km = fit_dictionary_knowledge(train_embeddings, dk_cfg)
        
            dict_R = km.transform_codes(train_embeddings)
            dict_active_sets = km.active_atoms(dict_R)
            dict_atom_freq = atom_frequencies(dict_active_sets, n_atoms=dk_cfg.n_components)
            row_norms = np.linalg.norm(dict_R, axis=1)
            print("R norms:", np.min(row_norms), np.median(row_norms), np.max(row_norms))

            # sparsity
            nnz = np.sum(np.abs(dict_R) > 1e-6, axis=1)
            print("R nnz:", np.min(nnz), np.median(nnz), np.max(nnz))

            # IMPORTANT: normalize to prevent norm artifacts dominating distances
            dict_R = dict_R / (np.linalg.norm(dict_R, axis=1, keepdims=True) + 1e-12)

            cluster_ids, eps = run_dbscan_thresholded(
                dict_R,
                k=args.dbscan_k,
                q=args.dbscan_q,      # try 0.05–0.15 for sparse codes
                min_samples=args.dbscan_min_samples,
                verbose=True,
            )

            # map noise -1 to singleton clusters (each noise point gets its own cluster ID)
            n_noise = int(np.sum(cluster_ids == -1))
            if n_noise > 0:
                max_cluster_id = int(np.max(cluster_ids[cluster_ids >= 0])) if np.any(cluster_ids >= 0) else -1
                cluster_ids = cluster_ids.copy()
                # Assign each noise point a unique cluster ID (singleton clusters)
                noise_indices = np.where(cluster_ids == -1)[0]
                for i, noise_idx in enumerate(noise_indices):
                    cluster_ids[noise_idx] = max_cluster_id + 1 + i
                logging.info(f"Assigned {n_noise} noise points to {n_noise} singleton clusters (IDs: {max_cluster_id + 1} to {max_cluster_id + n_noise})")
        
            if len(cluster_ids) != len(train_inputs):
                raise ValueError(f"Mismatch: cluster_ids={len(cluster_ids)} vs train_inputs={len(train_inputs)}")
        
            n_clusters = int(len(np.unique(cluster_ids)))
            logging.info(f"Run {run_idx + 1}: Dict+DBSCAN clustering produced {n_clusters} clusters (eps={eps:.4f})")
            plot_cluster_distribution(cluster_ids, output_dir, run_idx)
        
            # record cluster count
            if len(cluster_counts_per_run) <= run_idx:
                cluster_counts_per_run.append(n_clusters)
            else:
                cluster_counts_per_run[run_idx] = n_clusters
        
        else:
            raise ValueError(f"Unknown clustering method: {args.clustering}")
        
        # -------------------------
        # 2) Selection → produce demo indices
        # -------------------------
        sel_map: Dict[str, List[int]] = {}
        
    
        if "random" in args.baselines:
            sel_map["baseline_random"] = run_rng.sample(list(range(len(train_inputs))), budget)
        
        openicl_selections: Dict[str, Union[List[int], List[List[int]]]] = {}
        openicl_requested = [b for b in args.baselines if b in {"dpp", "mdl", "votek", "dpp_sgt", "mdl_sgt", "votek_sgt"}]
        if openicl_requested:
            train_targets_openicl = [label_names[y] for y in train_targets]
            test_targets_openicl = [label_names[y] for y in test_targets]
            dataset_reader = prepare_dataset_for_openicl(
                train_inputs=train_inputs,
                train_targets=train_targets_openicl,
                test_inputs=test_inputs,
                test_targets=test_targets_openicl,
            )
        
            # Aggressive memory clearing before retrievers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
        
            openicl_selections = run_openicl_retrievers(
                dataset_reader=dataset_reader,
                retrievers_to_run=openicl_requested,
                ice_num=budget,
                sentence_transformers_model=args.sentence_transformers_model,
                candidate_num=args.candidate_num,
                votek_k=args.votek_k,
                mdl_ce_model=args.mdl_ce_model,
                mdl_select_time=args.mdl_select_time,
                seed=run_seed,
                cluster_ids=cluster_ids,
                args=args,
            )
        
            # Aggressive GPU cache clearing after retrievers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
        
            # Query-specific retrievers return List[List[int]], corpus-level return List[int]
            query_specific_retrievers = {"dpp", "mdl", "dpp_sgt", "mdl_sgt"}
            for k, v in openicl_selections.items():
                if v is None:
                    continue
                if k in query_specific_retrievers:
                    # Per-query selections: v is List[List[int]]
                    # Validate and pad each query's selection
                    validated_per_query = []
                    for query_idx, query_sel in enumerate(v):
                        validated_sel = _validate_and_pad(query_sel, pool_size=len(train_inputs), k=budget, rng=run_rng)
                        validated_per_query.append(validated_sel)
                    sel_map[f"baseline_openicl_{k}"] = validated_per_query
                else:
                    # Corpus-level: v is List[int], validate and pad
                    sel = _validate_and_pad(v, pool_size=len(train_inputs), k=budget, rng=run_rng)
                    sel_map[f"baseline_openicl_{k}"] = sel
        
        # -------------------------
        # 3) Compute selection statistics (cluster info for each selected demo)
        # -------------------------
        def compute_selection_stats(
            selected_indices: List[int],
            cluster_ids: np.ndarray,
            method_name: str,
        ) -> Dict:
            """
            Compute cluster information for each selected demo.
            
            Returns:
                Dictionary with:
                - selected_demos: List of dicts, each with:
                  - index: relative index in pool
                  - cluster_id: cluster ID this demo belongs to
                  - cluster_size: size of that cluster
                - cluster_summary: Dict mapping cluster_id -> count of selected demos from that cluster
            """
            stats = {
                "method_name": method_name,
                "selected_demos": [],
                "cluster_summary": {},
            }
            
            # Compute cluster sizes for all clusters
            unique_clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
            cluster_size_map = dict(zip(unique_clusters, cluster_counts))
            
            # For each selected index, get its cluster info
            for idx in selected_indices:
                if idx < 0 or idx >= len(cluster_ids):
                    # Invalid index (shouldn't happen after validation, but handle gracefully)
                    continue
                
                cluster_id = int(cluster_ids[idx])
                cluster_size = int(cluster_size_map.get(cluster_id, 0))
                
                stats["selected_demos"].append({
                    "index": int(idx),
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                })
                
                # Update cluster summary
                if cluster_id not in stats["cluster_summary"]:
                    stats["cluster_summary"][cluster_id] = 0
                stats["cluster_summary"][cluster_id] += 1
            
            return stats
        
        # Compute selection stats for all methods (openicl baselines that use clustering)
        selection_stats = {}
        # Include both our methods and openicl baselines
        methods_to_stats = [
            name for name in sel_map.keys() 
            if name.startswith("baseline_openicl_")
        ]
        
        for method_name in methods_to_stats:
            if method_name not in sel_map:
                continue
                
            selected_indices = sel_map[method_name]
            
            # Handle query-specific retrievers (List[List[int]]) by aggregating across queries
            if isinstance(selected_indices, list) and len(selected_indices) > 0 and isinstance(selected_indices[0], list):
                # Per-query selection: aggregate all indices across queries
                aggregated_indices = []
                for query_sel in selected_indices:
                    aggregated_indices.extend(query_sel)
                # Get unique indices to avoid double-counting
                aggregated_indices = list(set(aggregated_indices))
                selected_indices = aggregated_indices
            
            # Compute stats (selected_indices is now List[int])
            selection_stats[method_name] = compute_selection_stats(
                selected_indices=selected_indices,
                cluster_ids=cluster_ids,
                method_name=method_name,
            )
        
        # Save selection stats for this run
        selection_stats_file = output_dir / f"selection_stats_run_{run_idx + 1}.json"
        with open(selection_stats_file, "w") as f:
            json.dump(selection_stats, f, indent=2)
        logging.info(f"Saved selection stats to {selection_stats_file}")
        
        # -------------------------
        # 4) Evaluate everything we produced this run
        # -------------------------
        # Initialize result lists if not present (so summary code doesn't break)
        for key in sel_map.keys():
            if key not in all_results:
                all_results[key] = []
        
        for name, indices in sel_map.items():
            # Store selected indices for analysis (relative to pool)
            if name not in all_selected_indices:
                all_selected_indices[name] = []
            all_selected_indices[name].append(indices.copy())
            
            try:
                acc = eval_selection(indices, name)
                all_results[name].append(acc)
            except Exception as e:
                logging.error(f"Eval failed for {name}: {e}")
                all_results[name].append(None)

    # Compute statistics (filter out None values)
    summary = {}
    for method, accs in all_results.items():
        # Filter out None values (from skipped retrievers)
        valid_accs = [x for x in accs if x is not None]
        if len(valid_accs) > 0:
            summary[method] = {
                "mean": float(np.mean(valid_accs)),
                "std": float(np.std(valid_accs)),
                "values": [float(x) for x in valid_accs],
                "n_runs": len(valid_accs),
            }
            logging.info(f"{method}: {summary[method]['mean']:.4f} ± {summary[method]['std']:.4f} (n={len(valid_accs)})")
        else:
            summary[method] = {
                "mean": None,
                "std": None,
                "values": [],
                "n_runs": 0,
            }
            logging.warning(f"{method}: No valid results (all runs failed or skipped)")

    # Compute cluster count statistics
    valid_cluster_counts = [x for x in cluster_counts_per_run if x is not None]
    if valid_cluster_counts:
        cluster_stats = {
            "mean": float(np.mean(valid_cluster_counts)),
            "std": float(np.std(valid_cluster_counts)),
            "min": int(np.min(valid_cluster_counts)),
            "max": int(np.max(valid_cluster_counts)),
            "values": [int(x) for x in valid_cluster_counts],
        }
        logging.info(
            f"Cluster counts: mean={cluster_stats['mean']:.1f} ± {cluster_stats['std']:.1f}, "
            f"range=[{cluster_stats['min']}, {cluster_stats['max']}]"
        )
        # Create aggregate plot across all runs
        plot_aggregate_cluster_statistics(cluster_counts_per_run, output_dir)
    else:
        cluster_stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "values": [],
        }

    metrics = {
        "eval_backend": eval_backend,
        "eval_model_name": eval_model_name if eval_backend == "hf" else endpoint_name,
        "endpoint_name": endpoint_name,  # Keep for backward compatibility
        "embedding_model_name": embedding_model_name,
        "layer": args.layer,
        "pooling": args.pooling,
        "l2_normalize": args.l2_normalize,
        "dataset_type": dataset_type,
        "budget": budget,
        "test_size": actual_test_size,  # actual test set size used
        "train_pool_size": train_pool_size,
        "n_runs": n_runs,
        "seed": seed,
        "sgt_lambda": args.sgt_lambda,
        "sgt_score_type": args.sgt_score_type,
        "dbscan_q": args.dbscan_q,
        "baselines": args.baselines,
        "clustering": args.clustering,
        "selection": args.selection,
        "cluster_counts": cluster_stats,
        "summary": summary,
        "all_results": all_results,
        "selected_indices": all_selected_indices,  # method -> list of indices per run (relative to pool)
        "pool_indices": pool_indices,  # pool indices (same for all runs, full training set)
    }
    
    # Add HF-specific config if using HF backend
    if eval_backend == "hf":
        metrics["hf_config"] = {
            "device": hf_device,
            "dtype": args.hf_dtype,
            "max_input_tokens": args.hf_max_input_tokens,
            "max_new_tokens": args.hf_max_new_tokens if args.hf_max_new_tokens is not None else args.max_tokens,
            "temperature": args.hf_temperature,
            "top_p": args.hf_top_p,
        }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_dir / 'metrics.json'}")
    logging.info(f"Metrics saved to {output_dir / 'metrics.json'}")
    
    # Also save selected indices as separate numpy files for easier access
    for method_name, indices_per_run in all_selected_indices.items():
        for run_idx, indices in enumerate(indices_per_run):
            indices_file = output_dir / f"selected_indices_{method_name}_run_{run_idx + 1}.npy"
            np.save(indices_file, np.array(indices))
    logging.info(f"Saved selected indices to {output_dir}")
    
    logging.info("Comparison finished.")


if __name__ == "__main__":
    main()

