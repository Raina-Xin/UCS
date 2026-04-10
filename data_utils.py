# src/data_utils.py

from typing import List, Tuple, Optional
from pathlib import Path
import json
from datasets import load_dataset


def load_banking77(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load the PolyAI/banking77 dataset and return (texts, labels).
    """
    ds = load_dataset("PolyAI/banking77", split=split, revision="refs/convert/parquet")
    texts = [ex["text"] for ex in ds]
    labels = [int(ex["label"]) for ex in ds]
    return texts, labels


def load_banking77_label_names() -> List[str]:
    """
    Returns the list of 77 intent names for Banking77.
    """
    ds = load_dataset("PolyAI/banking77", split="train", revision="refs/convert/parquet")
    label_names = ds.features["label"].names
    return label_names


def load_clinc150(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load the DeepPavlov/clinc150 dataset and return (texts, labels).
    """
    ds = load_dataset("DeepPavlov/clinc150", split=split)
    
    # Some labels are N/A
    ds = ds.filter(lambda x: x["label"] is not None)
    
    texts = [ex["utterance"] for ex in ds]
    labels = [int(ex["label"]) for ex in ds]
    return texts, labels


def load_clinc150_label_names() -> List[str]:
    """
    Returns the list of 150 intent names for CLINC150, ordered by their ID.
    """
    # Load 'intents' subset which contains the mapping
    intents_ds = load_dataset("DeepPavlov/clinc150", name="intents", split="intents")
    
    sorted_intents = intents_ds.sort("id")
    
    return sorted_intents["name"]


def load_hwu64(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load the DeepPavlov/hwu64 dataset and return (texts, labels).
    """
    ds = load_dataset("DeepPavlov/hwu64", split=split)
    
    # Some labels are N/A
    ds = ds.filter(lambda x: x["label"] is not None)
    
    texts = [ex["utterance"] for ex in ds]
    labels = [int(ex["label"]) for ex in ds]
    return texts, labels


def load_hwu64_label_names() -> List[str]:
    """
    Returns the list of 64 intent names for HWU64, ordered by their ID.
    """
    # Load 'intents' subset which contains the mapping
    intents_ds = load_dataset("DeepPavlov/hwu64", name="intents", split="intents")
    
    sorted_intents = intents_ds.sort("id")
    
    return sorted_intents["name"]


def load_bbh_task(task_name: str, data_dir: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """
    Load a BigBench Hard (BBH) task and return (inputs, targets).
    
    Args:
        task_name: Name of the BBH task (e.g., "boolean_expressions", "object_counting")
        data_dir: Directory containing BBH data. If None, uses default "big_bench_hard/bbh"
    
    Returns:
        inputs: List of input strings (questions)
        targets: List of target strings (answers)
    """
    if data_dir is None:
        # Default to big_bench_hard/bbh relative to this file
        data_dir = Path(__file__).parent / "big_bench_hard" / "bbh"
    else:
        data_dir = Path(data_dir)
    
    json_file = data_dir / f"{task_name}.json"
    if not json_file.exists():
        raise FileNotFoundError(f"BBH task file not found: {json_file}")
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    examples = data.get("examples", [])
    inputs = [ex["input"] for ex in examples]
    targets = [ex["target"] for ex in examples]
    
    return inputs, targets


def load_bbh_cot_prompt(task_name: str, data_dir: Optional[Path] = None) -> str:
    """
    Load the chain-of-thought prompt for a BBH task.
    
    Args:
        task_name: Name of the BBH task
        data_dir: Directory containing BBH CoT prompts. If None, uses default "big_bench_hard/cot-prompts"
    
    Returns:
        The CoT prompt text (including examples)
    """
    if data_dir is None:
        # Default to big_bench_hard/cot-prompts relative to this file
        data_dir = Path(__file__).parent / "big_bench_hard" / "cot-prompts"
    else:
        data_dir = Path(data_dir)
    
    txt_file = data_dir / f"{task_name}.txt"
    if not txt_file.exists():
        raise FileNotFoundError(f"BBH CoT prompt file not found: {txt_file}")
    
    with open(txt_file, "r") as f:
        prompt = f.read()
    
    return prompt


def list_available_bbh_tasks(data_dir: Optional[Path] = None) -> List[str]:
    """
    List all available BBH task names.

    Args:
        data_dir: Directory containing BBH data. If None, uses default "big_bench_hard/bbh"

    Returns:
        List of task names
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "big_bench_hard" / "bbh"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    tasks = []
    for json_file in data_dir.glob("*.json"):
        if json_file.name != "README.md":  # Skip README if it exists
            tasks.append(json_file.stem)

    return sorted(tasks)


def load_bbeh_task(task_name: str, data_dir: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """
    Load a BigBench Extra Hard (BBEH) task and return (inputs, targets).

    Expected layout:
      icl_select/bbeh/benchmark_tasks/<task_name>/task.json
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "bbeh" / "benchmark_tasks"
    else:
        data_dir = Path(data_dir)

    task_file = data_dir / task_name / "task.json"
    if not task_file.exists():
        raise FileNotFoundError(
            f"BBEH task file not found: {task_file}\n"
            f"Expected path: <data_dir>/{task_name}/task.json"
        )

    with open(task_file, "r") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    if not isinstance(examples, list) or len(examples) == 0:
        raise ValueError(f"BBEH task has no examples: {task_file}")

    inputs = [str(ex["input"]) for ex in examples]
    targets = [str(ex["target"]) for ex in examples]
    return inputs, targets


def list_available_bbeh_tasks(data_dir: Optional[Path] = None) -> List[str]:
    """
    List available BBEH task names from benchmark_tasks directory.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "bbeh" / "benchmark_tasks"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    tasks: List[str] = []
    for task_dir in sorted(data_dir.iterdir()):
        if task_dir.is_dir() and (task_dir / "task.json").exists():
            tasks.append(task_dir.name)
    return tasks
