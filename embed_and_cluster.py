# src/embed_and_cluster.py

from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- add near the top ---
import re
from typing import Dict

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

def _is_sharded(model) -> bool:
    return hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict) and len(model.hf_device_map) > 1

def _is_sentence_transformer_model(model_name: str) -> bool:
    """
    Check if a model name indicates it's a sentence-transformers model.
    
    Args:
        model_name: Model name or path
        
    Returns:
        True if the model appears to be a sentence-transformers model
    """
    model_name_lower = model_name.lower()
    # Check for common sentence-transformers indicators
    return (
        "sentence-transformers" in model_name_lower or
        "sentence_transformers" in model_name_lower or
        "sbert" in model_name_lower or
        model_name_lower.startswith("all-") or  # e.g., all-mpnet-base-v2
        model_name_lower.startswith("paraphrase-") or
        model_name_lower.startswith("multi-qa-")
    )


def _pick_input_device_from_map(hf_device_map: Dict[str, str]) -> torch.device:
    """
    Pick a reasonable device for input tensors when model is sharded.
    Prefer the embedding layer device; otherwise fall back to first device in map.
    """
    # Common embedding keys across architectures
    preferred_keys = [
        "model.embed_tokens",
        "transformer.wte",
        "gpt_neox.embed_in",
        "model.decoder.embed_tokens",
        "model.embeddings.word_embeddings",
    ]
    for k in preferred_keys:
        if k in hf_device_map:
            return torch.device(hf_device_map[k])

    # Fallback: first device in map
    first_dev = next(iter(hf_device_map.values()))
    return torch.device(first_dev)


class LLMEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        layer: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float16,
        device_map: Optional[str] = "auto",  # <-- make configurable
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_name = model_name
        
        # Qwen2.5 models may have numerical instability with float16
        # Prefer bfloat16 if available (more stable than float16), else use float32
        if "qwen" in model_name.lower() and torch_dtype == torch.float16:
            import logging
            if self.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                logging.info(
                    f"Qwen model detected: {model_name}. "
                    f"Using bfloat16 instead of float16 to avoid numerical instability."
                )
                torch_dtype = torch.bfloat16
            else:
                logging.warning(
                    f"Qwen model detected: {model_name}. "
                    f"Using float32 instead of float16 to avoid numerical instability."
                )
                torch_dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # If NOT sharded, move model to single device explicitly
        self.sharded = _is_sharded(self.model)
        if not self.sharded:
            self.model.to(self.device)

        # Determine layer
        if layer is None:
            num_layers = getattr(self.model.config, "num_hidden_layers", None)
            if num_layers is None:
                # fallback heuristics
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    num_layers = len(self.model.model.layers)
                elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                    num_layers = len(self.model.transformer.h)
                else:
                    num_layers = 32
            self.layer = num_layers // 2
        else:
            self.layer = layer

        self.model.eval()


    @torch.no_grad()
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 256,
        pooling: str = "mean",  # mean | first | last
        l2_normalize: bool = True,
    ) -> np.ndarray:
        all_embs = []

        # pick the correct device for inputs
        if self.sharded:
            input_device = _pick_input_device_from_map(self.model.hf_device_map)
        else:
            input_device = torch.device(self.device)

        for i in tqdm(range(0, len(texts), batch_size), desc="LLM Embedding"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            # move ONLY inputs (model already placed correctly)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            # Disable cache for models that use HybridCache (like Gemma-2) to avoid torch._dynamo issues
            # This is safe for embedding extraction as we don't need caching
            use_cache = not ("gemma" in self.model_name.lower() or "gemma2" in self.model_name.lower())
            
            outputs = self.model(**inputs, use_cache=use_cache)

            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError(
                    f"Model {self.model_name} did not return hidden_states. "
                    f"Make sure output_hidden_states=True when loading the model."
                )

            if self.layer >= len(outputs.hidden_states):
                raise RuntimeError(
                    f"Requested layer {self.layer} but model only has {len(outputs.hidden_states)} layers "
                    f"(0..{len(outputs.hidden_states)-1})."
                )

            hidden = outputs.hidden_states[self.layer]  # (B, T, D)
            
            # Check for NaN in hidden states (debugging)
            if torch.any(torch.isnan(hidden)):
                import logging
                n_nan = torch.sum(torch.isnan(hidden)).item()
                total = hidden.numel()
                logging.error(
                    f"NaN detected in hidden states from {self.model_name} at layer {self.layer}: "
                    f"{n_nan}/{total} values are NaN. This may indicate model instability or numerical issues."
                )
                # Try to replace NaN with zeros to continue
                hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)

            if pooling == "mean":
                # masked mean over tokens
                attn = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
                denom = attn.sum(dim=1).clamp_min(1.0)
                emb = (hidden * attn).sum(dim=1) / denom
                
                # Check for NaN after pooling
                if torch.any(torch.isnan(emb)):
                    import logging
                    logging.error(
                        f"NaN detected after pooling for {self.model_name}. "
                        f"Hidden stats: min={hidden.min().item():.4f}, max={hidden.max().item():.4f}, "
                        f"mean={hidden.mean().item():.4f}, std={hidden.std().item():.4f}"
                    )
                    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            elif pooling == "first":
                emb = hidden[:, 0, :]
            elif pooling == "last":
                # last *non-padding* token per sequence
                lengths = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
                emb = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            all_embs.append(emb.detach().float().cpu().numpy())

        embs = np.vstack(all_embs)

        if l2_normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms

        return embs




def _make_cache_path(
    cache_root: Union[str, Path],
    dataset_name: str,
    model_name: str,
    layer: Optional[int],
    pooling: str,
    l2_normalize: bool,
    is_sentence_transformer: bool = False,
) -> Path:
    """
    Build a unique cache path for embeddings, based on dataset + model + config.
    
    Args:
        is_sentence_transformer: If True, layer is ignored (sentence-transformers don't have layers)
    """
    cache_root = Path(cache_root)
    dataset_dir = cache_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Make model name filesystem-safe
    safe_model = model_name.replace("/", "__")
    
    if is_sentence_transformer:
        # Sentence-transformers don't have layers, so we don't include layer in filename
        filename = f"{safe_model}_st_pool-{pooling}_l2-{int(l2_normalize)}.npy"
    else:
        # Handle layer: None = "middle", -1 = "last", or specific number
        if layer is None:
            layer_str = "middle"
        elif layer == -1:
            layer_str = "last"
        else:
            layer_str = str(layer)
        filename = f"{safe_model}_layer{layer_str}_pool-{pooling}_l2-{int(l2_normalize)}.npy"
    return dataset_dir / filename


def get_or_compute_embeddings(
    texts: List[str],
    model_name: str,
    cache_root: Union[str, Path],
    dataset_name: str,
    layer: Optional[int] = None,  # None = use middle layer (default), -1 = last layer, or specific index
    batch_size: int = 4,
    max_length: int = 256,
    pooling: str = "mean",
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    Compute embeddings with either LLMEmbedder or SentenceTransformer, but cache them on disk so we don't
    recompute for the same (dataset, model, layer, pooling, l2_normalize).

    Automatically detects if model_name is a sentence-transformers model and uses the appropriate embedder.
    
    Embeddings are saved under:
        {cache_root}/{dataset_name}/{model_name+config}.npy
    """
    # Check if this is a sentence-transformers model
    is_sentence_transformer = _is_sentence_transformer_model(model_name)
    
    cache_path = _make_cache_path(
        cache_root=cache_root,
        dataset_name=dataset_name,
        model_name=model_name,
        layer=layer,
        pooling=pooling,
        l2_normalize=l2_normalize,
        is_sentence_transformer=is_sentence_transformer,
    )

    if cache_path.exists():
        return np.load(cache_path)

    # Use sentence-transformers if detected
    if is_sentence_transformer:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                f"sentence-transformers library not installed. "
                f"Install with: pip install sentence-transformers"
            )
        
        import logging
        logging.info(f"Using SentenceTransformer for model: {model_name}")
        
        # Load sentence-transformers model
        model = SentenceTransformer(model_name)
        
        # Encode texts in batches
        # SentenceTransformer.encode() handles batching internally, but we can also batch manually
        # Use normalize_embeddings parameter for L2 normalization
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="SentenceTransformer Embedding"):
            batch = texts[i : i + batch_size]
            batch_embs = model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=l2_normalize,  # SentenceTransformer has built-in L2 normalization
            )
            all_embs.append(batch_embs)
        
        embs = np.vstack(all_embs)
        
        # Ensure normalization if requested (SentenceTransformer should have done it, but double-check)
        if l2_normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
    else:
        # Use LLM embedder
        embedder = LLMEmbedder(
            model_name=model_name,
            layer=layer,  # None = middle layer, -1 = last layer, or specific index
        )
        embs = embedder.encode_batch(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
            l2_normalize=l2_normalize,
        )

    np.save(cache_path, embs)
    return embs


def compute_eps_from_knn(
    embeddings: np.ndarray,
    k: int = 10,
    q: float = 0.3,
    verbose: bool = True,
) -> float:
    """
    Compute DBSCAN eps as the q-quantile of k-th NN distances.
    
    Following the approach: compute top-k neighbor distances for each point,
    then aggregate all distances and set threshold to q-quantile of combined set.
    """
    n_samples = len(embeddings)
    n_neighbors = min(k + 1, n_samples)  # Can't ask for more neighbors than samples
    
    # use k+1 because the 0-th neighbor is the point itself
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, _ = knn.kneighbors(embeddings)
    
    # distances[:, 0] is 0 (self); take k-th neighbor (or last if k is too large)
    neighbor_idx = min(k, distances.shape[1] - 1)
    kth_dist = distances[:, neighbor_idx]
    
    # Filter out zero distances (can happen with duplicate points)
    kth_dist = kth_dist[kth_dist > 0]
    
    if len(kth_dist) == 0:
        # All distances are zero, return a small default
        return 1e-6
    
    if verbose:
        import logging
        logging.info(f"k-th ({k}) neighbor distances: min={kth_dist.min():.4f}, "
                    f"max={kth_dist.max():.4f}, mean={kth_dist.mean():.4f}, "
                    f"median={np.median(kth_dist):.4f}")
    
    eps = float(np.quantile(kth_dist, q))
    return eps


def run_dbscan_thresholded(
    embeddings: np.ndarray,
    k: int = 10,
    q: float = 0.3,
    min_samples: int = 3,
    min_eps: float = 1e-6,
    max_eps: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Cluster embeddings using DBSCAN with eps chosen from k-NN distances.
    Returns (cluster_ids, eps).
    
    Args:
        embeddings: Input embeddings
        k: Number of neighbors for k-NN
        q: Quantile for selecting eps
        min_samples: Minimum samples for DBSCAN
        min_eps: Minimum eps value to use (default 1e-6) to avoid eps=0 errors
        max_eps: Maximum eps value to use (default None, no limit). Useful for normalized vectors.
        verbose: Whether to log distance statistics
    """
    eps = compute_eps_from_knn(embeddings, k=k, q=q, verbose=verbose)
    
    # Ensure eps is greater than 0 (DBSCAN requirement)
    if eps <= 0:
        # If eps is 0 or negative, compute a fallback based on mean distance
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings))).fit(embeddings)
        distances, _ = knn.kneighbors(embeddings)
        # Use mean of k-th neighbor distances as fallback
        kth_dist = distances[:, k] if distances.shape[1] > k else distances[:, -1]
        eps = float(np.mean(kth_dist[kth_dist > 0])) if np.any(kth_dist > 0) else min_eps
        # Still ensure it's at least min_eps
        eps = max(eps, min_eps)
    
    # Cap eps if max_eps is specified (useful for normalized vectors)
    if max_eps is not None and eps > max_eps:
        if verbose:
            import logging
            logging.info(f"Capping eps from {eps:.4f} to {max_eps:.4f}")
        eps = max_eps
    
    if verbose:
        import logging
        logging.info(f"Using eps={eps:.4f} for DBSCAN (q={q} quantile of {k}-th neighbor distances)")
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = db.labels_.astype(int)
    return labels, eps
