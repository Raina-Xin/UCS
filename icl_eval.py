# src/icl_eval.py

from typing import List, Optional
import torch
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def clean_generation(text: str) -> str:
    """
    Clean up model output:
    - keep only the part after the last "Label:"
    - take only first line
    - strip whitespace
    - remove common prefixes/suffixes
    """
    text = text.strip()

    # Use the **last** "Label:" – that's where the model answers
    if "Label:" in text:
        text = text.split("Label:")[-1].strip()

    # Only the first line (avoid rambling)
    text = text.split("\n")[0].strip()
    
    # Remove common prefixes that models sometimes add
    prefixes_to_remove = ["the label is", "label is", "the intent is", "intent is", "answer:"]
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            # Remove colon if present
            if text.startswith(":"):
                text = text[1:].strip()
            break
    
    return text


def best_label_match(
    output: str,
    label_names: List[str],
) -> Optional[int]:
    """
    Robust label matcher:
    - exact match (case-insensitive, after normalization)
    - substring match (both directions)
    - token-overlap fallback
    """
    # Normalize: lowercase, strip, remove extra whitespace
    out = " ".join(output.lower().split())
    
    # 1. exact match (case-insensitive)
    for i, name in enumerate(label_names):
        name_normalized = " ".join(name.lower().split())
        if out == name_normalized:
            return i

    # 2. substring match (check both directions)
    for i, name in enumerate(label_names):
        name_lower = name.lower()
        # Check if label name is in output
        if name_lower in out:
            return i
        # Check if output is in label name (for partial matches)
        if out in name_lower and len(out) >= 3:  # Require at least 3 chars
            return i

    # 3. token overlap (improved: require majority of tokens)
    out_tokens = set(out.split())
    if len(out_tokens) == 0:
        return None
    
    best_score = 0
    best_idx = None
    for i, name in enumerate(label_names):
        name_tokens = set(name.lower().split())
        overlap = len(out_tokens & name_tokens)
        # Require at least 50% of output tokens to match
        if overlap > 0 and overlap >= len(out_tokens) * 0.5:
            if overlap > best_score:
                best_score = overlap
                best_idx = i

    return best_idx if best_score > 0 else None


def build_demo_block_banking77(texts, labels, label_names):
    """Build demo block for Banking77 classification task."""
    # Include the full label list to help the model understand available options
    label_list_str = ", ".join(label_names)
    lines = [
        "You are an assistant for intent classification.\n"
        "Your task is to assign EXACTLY ONE label to each query.\n"
        "Answer STRICTLY in the format: 'Label: <intent_name>'.\n",
        f"Available labels: {label_list_str}\n",
        "Here are some labeled examples:\n",
    ]
    for t, y in zip(texts, labels):
        lines.append(f"Example:\nText: {t}\nLabel: {label_names[y]}\n")
    lines.append("\nNow answer the following question.\n")
    return "\n".join(lines)


def build_demo_block_clinc150(texts, labels, label_names):
    """Build demo block for CLINC150 intent classification task."""
    label_list_str = ", ".join(label_names)
    lines = [
        "You are an assistant for intent classification.\n"
        "Your task is to assign EXACTLY ONE label to each query.\n"
        "Answer STRICTLY in the format: 'Label: <intent_name>'.\n",
        f"Available labels: {label_list_str}\n",
        "Here are some labeled examples:\n",
    ]
    for t, y in zip(texts, labels):
        lines.append(f"Example:\nText: {t}\nLabel: {label_names[y]}\n")
    lines.append("\nNow answer the following question.\n")
    return "\n".join(lines)


def build_demo_block_hwu64(texts, labels, label_names):
    """Build demo block for HWU64 intent classification task."""
    label_list_str = ", ".join(label_names)
    lines = [
        "You are an assistant for intent classification.\n"
        "Your task is to assign EXACTLY ONE label to each query.\n"
        "Answer STRICTLY in the format: 'Label: <intent_name>'.\n",
        f"Available labels: {label_list_str}\n",
        "Here are some labeled examples:\n",
    ]
    for t, y in zip(texts, labels):
        lines.append(f"Example:\nText: {t}\nLabel: {label_names[y]}\n")
    lines.append("\nNow answer the following question.\n")
    return "\n".join(lines)


def build_demo_block_bbh(train_inputs, train_targets):
    """
    Build demo block for BBH tasks using few-shot format (similar to Banking77).
    
    Args:
        train_inputs: List of training input strings
        train_targets: List of training target strings (answers)
    """
    lines = [
        "You are an assistant for answering questions.\n"
        "Your task is to provide the correct answer.\n"
        "Answer STRICTLY in the format: 'A: <answer>'.\n",
        "Here are some examples:\n",
    ]
    for inp, tgt in zip(train_inputs, train_targets):
        lines.append(f"Q: {inp}\nA: {tgt}\n")
    lines.append("\nNow answer the following question.\n")
    return "\n".join(lines)


def build_prompt_banking77(demo_block: str, query: str):
    return demo_block + f"Text: {query}\nLabel:"


def build_prompt_clinc150(demo_block: str, query: str):
    return demo_block + f"Text: {query}\nLabel:"


def build_prompt_hwu64(demo_block: str, query: str):
    return demo_block + f"Text: {query}\nLabel:"


def build_prompt_bbh(demo_block: str, query: str):
    return demo_block + f" {query}\nA:"


def evaluate_bbh_answer(generated: str, target: str) -> bool:
    """
    Evaluate if generated answer matches target for BBH tasks.
    Handles:
    - True/False answers (boolean_expressions)
    - Multiple choice options like "(A)", "(B)", etc. (date_understanding)
    
    Uses flexible matching: exact match, normalized match, or contains target.
    """
    import re
    
    # Clean and normalize
    generated_clean = generated.strip()
    target_clean = target.strip()
    
    # Normalize to lowercase for comparison (but preserve original for pattern matching)
    generated_lower = generated_clean.lower()
    target_lower = target_clean.lower()
    
    # 1. Exact match (case-insensitive)
    if generated_lower == target_lower:
        return True
    
    # 2. For multiple choice options like "(A)", "(B)", extract the letter
    # Target might be "(B)" or "B" or "(B)" - handle all cases
    target_letter = None
    if target_clean.startswith("(") and target_clean.endswith(")"):
        target_letter = target_clean[1:-1].strip().upper()
    elif len(target_clean) == 1 and target_clean.isalpha():
        target_letter = target_clean.upper()
    
    if target_letter:
        # Look for the letter in parentheses in generated text
        # Pattern: (A), (B), etc.
        pattern = r"\(([A-Z])\)"
        matches = re.findall(pattern, generated_clean.upper())
        if matches and matches[-1] == target_letter:  # Use last match (most likely the answer)
            return True
        # Also check if just the letter appears (without parentheses)
        if target_letter in generated_clean.upper():
            # Make sure it's not part of a word
            pattern_word = r"\b" + target_letter + r"\b"
            if re.search(pattern_word, generated_clean.upper()):
                return True
    
    # 3. For True/False answers
    if target_lower in ["true", "false"]:
        # Check if generated contains the target (case-insensitive)
        if target_lower in generated_lower:
            # Make sure it's not part of another word
            pattern_word = r"\b" + target_lower + r"\b"
            if re.search(pattern_word, generated_lower):
                return True
    
    # 4. Check if target is in generated (for cases where model adds extra text)
    if target_lower in generated_lower:
        return True
    
    # 5. Try to extract final answer (look for patterns like "So the answer is X" or "answer: X")
    patterns = [
        r"so the answer is\s+([^\s\.]+)",
        r"answer is\s+([^\s\.]+)",
        r"answer:\s*([^\s\.]+)",
        r"the answer is\s+([^\s\.]+)",
        r"a:\s*([^\s\.]+)",  # For "A: True" format
    ]
    for pattern in patterns:
        match = re.search(pattern, generated_lower)
        if match:
            extracted = match.group(1).strip()
            # Remove parentheses if present
            if extracted.startswith("(") and extracted.endswith(")"):
                extracted = extracted[1:-1].strip()
            if extracted == target_lower or target_lower in extracted:
                return True
            # For multiple choice, check if extracted letter matches
            if target_letter and extracted.upper() == target_letter:
                return True
    
    return False


def run_icl_eval(
    model_name: str,
    train_texts: Optional[List[str]] = None,
    train_labels: Optional[List[int]] = None,
    test_texts: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    label_names: Optional[List[str]] = None,
    # BBH format
    train_inputs: Optional[List[str]] = None,
    train_targets: Optional[List[str]] = None,
    test_inputs: Optional[List[str]] = None,
    test_targets: Optional[List[str]] = None,
    dataset_type: str = "banking77",  # "banking77", "clinc150", "hwu64", or "bbh"
    max_new_tokens: int = 16,
    device: str = "cuda",
    model=None,
    tokenizer=None,
    batch_size: int = 8,
) -> float:
    """
    Run ICL evaluation with batching for speed. If model and tokenizer are provided, 
    they will be reused instead of loading from scratch (useful for multiple evaluations).
    
    IMPORTANT: Always pass model and tokenizer if you've already loaded them to avoid
    loading multiple models and causing OOM errors.
    
    Args:
        dataset_type: "banking77", "clinc150", "hwu64", or "bbh"
        batch_size: Number of examples to process in parallel (default 8)
    """
    # Validate inputs based on dataset type
    if dataset_type == "banking77" or dataset_type == "clinc150" or dataset_type == "hwu64":
        if train_texts is None or train_labels is None or test_texts is None or test_labels is None or label_names is None:
            raise ValueError(f"For {dataset_type}, train_texts, train_labels, test_texts, test_labels, and label_names are required")
    elif dataset_type == "bbh":
        if train_inputs is None or train_targets is None or test_inputs is None or test_targets is None:
            raise ValueError("For BBH, train_inputs, train_targets, test_inputs, and test_targets are required")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'banking77', 'clinc150', 'hwu64', or 'bbh'")
    
    # Only load if not provided - this prevents loading multiple models
    if tokenizer is None:
        logging.warning("Tokenizer not provided, loading from scratch. This may cause OOM if model is also not provided.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set pad_token if it doesn't exist (some models like Mistral don't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    if model is None:
        logging.warning("Model not provided, loading from scratch. This may cause OOM if a model is already loaded!")
        # Clear cache before loading to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()

    # Build demo block based on dataset type
    if dataset_type == "banking77":
        demo_block = build_demo_block_banking77(train_texts, train_labels, label_names)
        test_queries = test_texts
        test_answers = [label_names[y] for y in test_labels]
        build_prompt_fn = build_prompt_banking77
        evaluate_fn = lambda gen, ans: best_label_match(clean_generation(gen), label_names) == test_labels[test_queries.index(test_texts[test_queries.index(gen)])] if gen in test_queries else False
        # Better evaluation for banking77
        def eval_banking77(gen, true_idx):
            pred_idx = best_label_match(clean_generation(gen), label_names)
            return pred_idx == true_idx
        evaluate_fn = eval_banking77
    elif dataset_type == "clinc150":
        demo_block = build_demo_block_clinc150(train_texts, train_labels, label_names)
        test_queries = test_texts
        test_answers = [label_names[y] for y in test_labels]
        build_prompt_fn = build_prompt_clinc150
        
        def eval_clinc150(gen, true_idx):
            pred_idx = best_label_match(clean_generation(gen), label_names)
            return pred_idx == true_idx
        evaluate_fn = eval_clinc150
    elif dataset_type == "hwu64":
        demo_block = build_demo_block_hwu64(train_texts, train_labels, label_names)
        test_queries = test_texts
        test_answers = [label_names[y] for y in test_labels]
        build_prompt_fn = build_prompt_hwu64
        
        def eval_hwu64(gen, true_idx):
            pred_idx = best_label_match(clean_generation(gen), label_names)
            return pred_idx == true_idx
        evaluate_fn = eval_hwu64
    else:  # bbh
        demo_block = build_demo_block_bbh(train_inputs, train_targets)
        test_queries = test_inputs
        test_answers = test_targets
        build_prompt_fn = build_prompt_bbh
        evaluate_fn = evaluate_bbh_answer
    
    correct = 0
    total = len(test_queries)
    failed_matches = []  # Store failed predictions for debugging
    
    # Check if demo block might be too long (warn about truncation)
    demo_tokens = tokenizer.encode(demo_block, add_special_tokens=False)
    if len(demo_tokens) > 2000:  # Rough threshold
        logging.warning(f"Demo block is long ({len(demo_tokens)} tokens). "
                       f"Prompts may be truncated if they exceed model's max length.")
        # Use print() to ensure it goes to stderr immediately
        print(f"\n{'='*80}", file=sys.stderr, flush=True)
        print(f"DEMO BLOCK ANALYSIS:", file=sys.stderr, flush=True)
        print(f"{'='*80}", file=sys.stderr, flush=True)
        print(f"Demo block length: {len(demo_block)} characters, {len(demo_tokens)} tokens", file=sys.stderr, flush=True)
        n_train = len(train_texts) if dataset_type in ["banking77", "clinc150", "hwu64"] else len(train_inputs)
        print(f"Number of training examples: {n_train}", file=sys.stderr, flush=True)
        print(f"\nDemo block preview (first 1000 chars):", file=sys.stderr, flush=True)
        print(f"{demo_block[:1000]}...", file=sys.stderr, flush=True)
        print(f"\nDemo block preview (last 1000 chars):", file=sys.stderr, flush=True)
        print(f"...{demo_block[-1000:]}", file=sys.stderr, flush=True)
        print(f"{'='*80}\n", file=sys.stderr, flush=True)

    # Process in batches for speed
    for batch_start in tqdm(
        range(0, len(test_queries), batch_size),
        desc="ICL Eval",
        total=(len(test_queries) + batch_size - 1) // batch_size,
    ):
        batch_end = min(batch_start + batch_size, len(test_queries))
        batch_queries = test_queries[batch_start:batch_end]
        batch_answers = test_answers[batch_start:batch_end]
        
        # For Banking77, CLINC150, and HWU64, also need true label indices
        if dataset_type == "banking77" or dataset_type == "clinc150" or dataset_type == "hwu64":
            batch_true_indices = test_labels[batch_start:batch_end]
        
        # Build prompts for this batch
        batch_prompts = [build_prompt_fn(demo_block, query) for query in batch_queries]
        
        # Tokenize with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Store prompt lengths for each example (using attention mask to exclude padding)
        prompt_lens = inputs["attention_mask"].sum(dim=1).cpu().tolist()
        
        # Check if input is too long (warn but continue)
        max_prompt_len = max(prompt_lens)
        if max_prompt_len > 2000:
            logging.warning(f"Long prompt detected ({max_prompt_len} tokens). This may cause OOM.")
            # Print the first prompt in the batch as an example - use print() to ensure it appears
            if batch_start == 0:  # Only print for first batch to avoid spam
                print(f"\n{'='*80}", file=sys.stderr, flush=True)
                print(f"EXAMPLE PROMPT (first in batch) - BEFORE GENERATION:", file=sys.stderr, flush=True)
                print(f"{'='*80}", file=sys.stderr, flush=True)
                print(f"Prompt length: {len(batch_prompts[0])} characters, {max_prompt_len} tokens", file=sys.stderr, flush=True)
                print(f"Query: {batch_queries[0][:200]}...", file=sys.stderr, flush=True)
                print(f"\nFull prompt:", file=sys.stderr, flush=True)
                print(f"{batch_prompts[0]}", file=sys.stderr, flush=True)
                print(f"{'='*80}\n", file=sys.stderr, flush=True)
                # Also log it
                logging.info(f"Full prompt printed to stderr (length: {len(batch_prompts[0])} chars, {max_prompt_len} tokens)")

        with torch.no_grad():
            try:
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id,
                )
            except torch.cuda.OutOfMemoryError as e:
                logging.error(f"OOM error during generation! Try reducing batch_size (current: {batch_size}) or max_new_tokens (current: {max_new_tokens})")
                # Clear cache and re-raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

        # Process each example in the batch
        for i in range(len(batch_queries)):
            prompt_len = prompt_lens[i]
            # Extract only the generated tokens (skip prompt)
            # gen[i] has shape [max_prompt_len_in_batch + max_new_tokens], 
            # but we only want the tokens after the actual prompt length
            continuation_ids = gen[i][prompt_len:].cpu()
            
            # Find where generation actually ends (eos token or padding)
            if tokenizer.eos_token_id is not None:
                eos_positions = (continuation_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    continuation_ids = continuation_ids[:eos_positions[0]]
            
            # Remove any remaining padding tokens
            continuation_ids = continuation_ids[continuation_ids != tokenizer.pad_token_id]
            
            continuation = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
            )

            # Evaluate based on dataset type
            if dataset_type == "banking77" or dataset_type == "clinc150" or dataset_type == "hwu64":
                cleaned = clean_generation(continuation)
                pred_idx = best_label_match(cleaned, label_names)
                true_idx = batch_true_indices[i]
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
                            "raw_output": continuation[:100],  # First 100 chars
                        })
            else:  # bbh
                target = batch_answers[i]
                is_correct = evaluate_bbh_answer(continuation, target)
                
                if is_correct:
                    correct += 1
                else:
                    # Log failed predictions for debugging (sample first 10)
                    if len(failed_matches) < 10:
                        failed_matches.append({
                            "true_answer": target,
                            "generated_output": continuation[:200],  # First 200 chars
                            "query": batch_queries[i][:100],
                        })
        
        # Clear cache after each batch to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Log summary of failures
    if failed_matches:
        logging.debug(f"Sample failed predictions (showing first {len(failed_matches)}):")
        for i, fail in enumerate(failed_matches[:5]):  # Show first 5
            if dataset_type == "banking77" or dataset_type == "clinc150" or dataset_type == "hwu64":
                logging.debug(f"  {i+1}. True: {fail.get('true_label', 'N/A')}, "
                             f"Pred: {fail.get('pred_label', 'N/A')}, "
                             f"Generated: '{fail.get('cleaned_output', 'N/A')}'")
            else:  # bbh
                logging.debug(f"  {i+1}. Query: {fail.get('query', 'N/A')[:50]}..., "
                             f"True: {fail.get('true_answer', 'N/A')}, "
                             f"Generated: '{fail.get('generated_output', 'N/A')[:50]}...'")
    
    accuracy = correct / total
    logging.info(f"ICL Evaluation: {correct}/{total} correct ({accuracy:.4f})")
    if len(failed_matches) > 0:
        logging.info(f"Failed to match {total - correct} predictions. "
                    f"Check debug logs for examples.")
    
    return accuracy
