# segment.py

# ===== Backends & globals =====
from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
from typing import List

_BACKEND: str = "st"          # "hf" or "st"
_TOKENIZER = None             # used for token counting in both backends
_MODEL = None                 # HF model (hf backend)
_ST_MODEL = None              # SentenceTransformer model (st backend)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MAX_TOKENS = 320 # our token budget;
"""
Eventually, had to bump it down from 512 to a saf*er* 320, because the tokenizer of the RST parser apparently tokenizes differently
and some resulting segments, though below the 512 budget with the tokenizer in this module (e.g. from the "sberbank-ai/sbert_large_nlu_ru" model),
exceeded the 512 limit downstream when processed with the RST parser. Through trial and error, landed on 320.
"""
_MIN_TOKENS = 200 # the minimum of tokens a chunk can be split into
_WINDOW = 1 # the size of the moving window in sentences to be compared to the previous chunk of _MAX_PREVIOUS size in sentences
_MAX_PREVIOUS = 3
_SPECIAL_OVERHEAD: Optional[int] = None # Special tokens returned by the model like CLS and EOS

def _require_model():
    if _BACKEND == "hf":
        if _TOKENIZER is None or _MODEL is None:
            raise RuntimeError("HF model not initialized. Call init_embeddings(backend='hf', model_name=...)")
    elif _BACKEND == "st":
        if _ST_MODEL is None or _TOKENIZER is None:
            raise RuntimeError("ST model not initialized. Call init_embeddings(backend='st', model_name=...)")
    else:
        raise RuntimeError(f"Unknown backend {_BACKEND!r}")

def _get_special_overhead() -> int:
    """
    Compute how many special tokens the current tokenizer adds to a single sequence.
    Cached after first call once. Works for both HF and ST backends.
    """
    global _SPECIAL_OVERHEAD
    if _SPECIAL_OVERHEAD is not None:
        return _SPECIAL_OVERHEAD

    _require_model()  # ensure tokenizer exists
    
    try:
        no_spec = _TOKENIZER("", add_special_tokens=False)["input_ids"]
        with_spec = _TOKENIZER("", add_special_tokens=True)["input_ids"]
        _SPECIAL_OVERHEAD = max(0, len(with_spec) - len(no_spec))
    except Exception:
        # Very defensive: if something odd happens, fall back to 2 (BERT-ish)
        _SPECIAL_OVERHEAD = 2
    return _SPECIAL_OVERHEAD

# --- Optional NLTK setup ---
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False
    sent_tokenize = None

# -----------------------------------------------------
# Callable / Public: explicitly inintializing the model
# -----------------------------------------------------
def init_embeddings(backend: str = _BACKEND, *, model_name: str) -> None:
    """
    Initialize either a HuggingFace encoder ('hf') or a SentenceTransformers model ('st').
    """
    global _BACKEND, _TOKENIZER, _MODEL, _ST_MODEL, _SPECIAL_OVERHEAD
    _BACKEND = backend
    _SPECIAL_OVERHEAD = None

    if not model_name:
        raise ValueError("model_name is required and must be a non-empty string")

    if backend == "hf":
        name = model_name
        _TOKENIZER = AutoTokenizer.from_pretrained(name)
        _MODEL = AutoModel.from_pretrained(name).to(_DEVICE).eval()
        _ST_MODEL = None

    elif backend == "st":
        from sentence_transformers import SentenceTransformer
        name = model_name
        _ST_MODEL = SentenceTransformer(name)
        
        # Use the ST model's underlying tokenizer for token counting
        _TOKENIZER = _ST_MODEL.tokenizer
        _MODEL = None

    else:
        raise ValueError("backend must be 'hf' or 'st'")

# === Helper to check if NLTK resources are in place
def _ensure_nltk(language: str = "russian") -> None:
    """
    Lazily fetch punkt resources if available.
    """
    if not _NLTK_OK:
        raise RuntimeError("NLTK is not available. Install nltk and download punk and/or punkt_tab.")
    
    # Newer NLTK versions separate 'punkt' and 'punkt_tab'.
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

# === Sentence splitting helper ===
def _split_into_sentences(text: str, language: str = "russian", splitter: Optional[Callable[[str], List[str]]] = None):
    if splitter:
        return splitter(text)
    
    try:
        _ensure_nltk(language)                     # raises if NLTK missing
        return sent_tokenize(text, language=language)
    except RuntimeError:
        # last resort; quality may be poor
        return [s.strip() for s in text.split(".") if s.strip()]


# === Token counting utilities ===
# === Count the tokens in a sentence after we have made sure the sentence *does not* exceed _MAX_TOKENS in BERT-style tokens
def _per_sentence_token_counts(sentences: List[str]) -> np.ndarray:
    """
    Token counts per sentence WITHOUT special tokens (add_special_tokens=False),
    so sums are additive across boundaries. We account for special tokens in the main splitter function.
    """
    _require_model()
    batch = _TOKENIZER(
        sentences,
        add_special_tokens=False,
        truncation=False
    )
    counts = [len(ids) for ids in batch["input_ids"]]
    return np.asarray(counts, dtype=np.int32)

# === Check the edge cases of one stretch of text including special tokens not fitting under the _MAX_TOKENS limit ===
def _text_fits_under_limit(text: str, limit: int) -> bool:
    
    _require_model()

    out = _TOKENIZER(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=limit,
        return_overflowing_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    # If more than one window, it definitely overflowed
    if "overflow_to_sample_mapping" in out and len(out["overflow_to_sample_mapping"]) > 1:
        return False

    # If reported length hit the cap, assume truncation
    if "length" in out and out["length"][0] >= limit:
        return False

    # Final fallback: check input_ids
    ids = out.get("input_ids")
    if ids and len(ids) >= limit:
        return False
    
    return True

# === Check if the "content" (whatever it happens to be) fits under the _MAX_TOKENS limit
# minus the special tokens -- these are accounted for when this function is called inside another function
# where special tokens are added on top and the sum is measure against the limit===
def _content_fits_under_limit(chunk: str, limit: int) -> bool:
    
    _require_model()

    out = _TOKENIZER(
        chunk,
        add_special_tokens=False,
        truncation=True,
        max_length=limit,
        return_overflowing_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    # If more than one window, it definitely overflowed
    if "overflow_to_sample_mapping" in out and len(out["overflow_to_sample_mapping"]) > 1:
        return False

    # If reported length hit the cap, assume truncation
    if "length" in out and out["length"][0] >= limit:
        return False

    # Final fallback: check input_ids
    ids = out.get("input_ids")
    if ids and len(ids) >= limit:
        return False
    
    return True

# === An extreme case of one 'word' esceeding the _MAX_LIMIT if we were to feed it into the tokenizer
def _split_word_by_token_limit(word: str, limit: int) -> list[str]:
    """Split a single oversize 'word' (e.g., very long URL) by token limit."""
    parts, start = [], 0
    
    while start < len(word):
        lo, hi = 1, len(word) - start
        best = 1
        
        # binary-search largest slice that fits the budget in tokens
        while lo <= hi:
            mid = (lo + hi) // 2
            piece = word[start:start+mid]
            if _content_fits_under_limit(piece, limit):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        piece = word[start:start+best]
        
        # guard against pathological cases where even 1 char > budget in tokens (_MAX_TOKENS)
        if not piece:
            # force split by tokenizer truncation to avoid infinite loop
            ids = _TOKENIZER(word[start:], add_special_tokens=False, truncation=True, max_length=limit)["input_ids"]
            piece = _TOKENIZER.decode(ids, skip_special_tokens=True)
            if not piece:  # extreme edge
                piece = word[start:start+1]
        parts.append(piece)
        start += len(piece)
    return parts

def _split_long_sentence(sentence: str, limit: int) -> List[str]:
    """
    Split a single overlong sentence into sub-chunks that each fit `limit`
    (measured in model tokens, excluding special tokens).
    """
    parts: List[str] = []
    cur = ""

    # Keep words and the whitespace between them, so spacing looks natural later
    tokens = re.findall(r"\S+|\s+", sentence)

    for tok in tokens:
        # 1) Whitespace: just accumulate it; we'll check the token budget when we add the next word
        if tok.isspace():
            cur += tok
            continue

        # 2) Try to append this word to the current chunk; check if it still fits
        candidate = (cur + tok) if cur else tok
        if _content_fits_under_limit(candidate, limit):
            cur = candidate
            continue

        # 3) Adding this word would overflow → finalize the current chunk (if it has content)
        if cur.strip():
            parts.append(cur.strip())
            cur = ""

        # 4) Now place the word itself. If the word alone still doesn't fit, split the word.
        if not _content_fits_under_limit(tok, limit):
            for piece in _split_word_by_token_limit(tok, limit):
                if _content_fits_under_limit(piece, limit):
                    parts.append(piece)
                else:
                    # Ultra-defensive last resort: force tokenizer truncation
                    ids = _TOKENIZER(
                        piece,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=limit
                    )["input_ids"]
                    parts.append(_TOKENIZER.decode(ids, skip_special_tokens=True))
        else:
            # 5) The word fits by itself: start a new chunk with it
            cur = tok

    # 6) Flush any trailing content
    if cur.strip():
        parts.append(cur.strip())

    return parts

# === Helper to get sentence embeddings (batched + masked mean-pooling for the HF backend) ===
@torch.inference_mode()
def _get_sentence_embeddings(sentences: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Return L2-normalized sentence embeddings as (N, H) numpy array.
    - HF backend: masked mean over last_hidden_state + normalize
    - ST backend: SentenceTransformer.encode(..., normalize_embeddings=True)
    """
    _require_model()

    if _BACKEND == "st":
        # Sentence Transformers does pooling & normalization on their own (Yay!)
        embs = _ST_MODEL.encode(
            sentences,
            batch_size=max(32, batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embs  # shape (N, H), already L2-normalized

    # HF path
    embs: List[np.ndarray] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = _TOKENIZER(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_MAX_TOKENS
        ).to(_DEVICE)

        outputs = _MODEL(**inputs)                      # last_hidden_state: (B, T, H)
        last = outputs.last_hidden_state                # (B, T, H)
        mask = inputs["attention_mask"].unsqueeze(-1)   # (B, T, 1)

        # masked mean
        summed = (last * mask).sum(dim=1)               # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)           # (B, 1)
        pooled = summed / counts                        # (B, H)

        # L2 normalize rows
        pooled = F.normalize(pooled, p=2, dim=1)        # (B, H)
        embs.append(pooled.detach().cpu().numpy())

    if not embs:
        # Choose the right H for empty result
        H = _MODEL.config.hidden_size
        return np.empty((0, H), dtype=np.float32)
    return np.vstack(embs)

# === Cosine similarity helper (cosine sim = dot product because vectors have been normalized)
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    assert a.ndim == b.ndim == 1, f"cosine_sim expects 1D, got {a.shape} and {b.shape}"
    return float(np.dot(a, b))

# ==== A Helper to pre-screen long sentences and split them
# so we can feed the list of sentences where no sentence exceeds the limit into the final splitter function
def _pre_split_rogues(sentences: List[str], max_tokens: int, debug: bool = False) -> List[str]:
    budget = max_tokens - _get_special_overhead()
    out: List[str] = []

    def _dbg(*args):
        if debug:
            print(*args)

    for idx, s in enumerate(sentences):
        if _content_fits_under_limit(s, budget):
            out.append(s)
        else:
            _dbg(f"[segment] rogue sentence at index {idx} — splitting")
            parts = _split_long_sentence(s, limit=budget)
            out.extend(p for p in (x.strip() for x in parts) if p)
    return out

# ==== Helper that splits text based on the similarity 'valley' between previous and window
def _multi_split_best_points(
    sentences: List[str],
    embeddings: np.ndarray,
    max_tokens: int,
    min_tokens: int,
    window_size: int,
    max_prev_sentences: int,
    *,
    debug: bool = False
) -> List[str]:
    """
    Split [sentences] into chunks under max_tokens (incl. specials),
    cutting at low-similarity “valleys”, with safeguards to avoid lost tails.
    Set debug=True to print the decisions.
    """
    if len(sentences) == 0:
        return []

    # === prefix sums for fast token-length checks
    token_counts = _per_sentence_token_counts(sentences)
    prefix = np.concatenate([[0], np.cumsum(token_counts)])

    def chunk_token_len(lo: int, hi: int) -> int:
        return int(prefix[hi] - prefix[lo] + _get_special_overhead())

    def _dbg(*args):
        if debug:
            print(*args)

    final_chunks: List[str] = []
    queue: List[Tuple[int, int]] = [(0, len(sentences))]

    while queue:
        lo, hi = queue.pop(0)

        # SAFEGUARD A: ignore empty spans
        if lo >= hi:
            _dbg(f"[skip-empty] span=({lo},{hi})")
            continue

        span_len = chunk_token_len(lo, hi)
        _dbg(f"[span] ({lo},{hi}) tokens≈{span_len}")

        # If the whole span fits → emit it.
        if span_len <= max_tokens:
            chunk = " ".join(sentences[lo:hi]).strip()
            if chunk:  # SAFEGUARD B: skip empties
                final_chunks.append(chunk)
                _dbg(f"  -> emit ({lo},{hi}) tokens≈{span_len}")
            else:
                _dbg(f"  -> emit skipped (empty after strip)")
            continue

        # Need to split
        best_idx = None
        best_sim = float("inf")

        # Try each candidate cut i
        start_i = lo + 1
        end_i   = hi - window_size + 1
        for i in range(start_i, end_i):
            before_len = chunk_token_len(lo, i)
            after_len  = chunk_token_len(i, hi)

            if before_len < min_tokens or after_len < min_tokens:
                _dbg(f"    [reject i={i}] before={before_len} after={after_len} (min={min_tokens})")
                continue

            left_start = max(lo, i - max_prev_sentences)
            left_emb   = embeddings[left_start:i].mean(axis=0)
            right_emb  = embeddings[i : i + window_size].mean(axis=0)
            sim = _cosine_sim(left_emb, right_emb)

            _dbg(f"    [i={i}] before={before_len} after={after_len} sim={sim:.4f}")

            if sim < best_sim:
                best_sim = sim
                best_idx = i

        # Fallback: midpoint if no legal candidate
        if best_idx is None:
            best_idx = (lo + hi) // 2
            _dbg(f"  [fallback] best_idx={best_idx} (midpoint)")

        # SAFEGUARD C: if split makes no progress, emit the span
        if best_idx <= lo or best_idx >= hi:
            chunk = " ".join(sentences[lo:hi]).strip()
            if chunk:
                final_chunks.append(chunk)
                _dbg(f"  -> emit (no-progress cut) ({lo},{hi})")
            else:
                _dbg(f"  -> emit skipped (no-progress but empty)")
            continue

        _dbg(f"  [split] best_idx={best_idx} best_sim={best_sim:.4f} -> ({lo},{best_idx}) + ({best_idx},{hi})")

        # Process left first: push right then left at the front
        queue.insert(0, (best_idx, hi))    # right
        queue.insert(0, (lo, best_idx))    # left

    return final_chunks

# -----------------------------------------------------------------------------
# Calllable / Public: Main function -- run a text through the whole pipeline
# -----------------------------------------------------------------------------
def segment_text(
    text: str,
    *,
    language: str = "russian",
    max_tokens: int = _MAX_TOKENS,
    min_tokens: int = _MIN_TOKENS,
    window_size: int = _WINDOW,
    max_prev_sentences: int = _MAX_PREVIOUS,
    batch_size: int = 32,
    splitter: Optional[Callable[[str], List[str]]] = None,
    debug: bool = False
) -> Tuple[List[str], List[str]]:
    """
    End-to-end: split into sentences -> embed -> multi-split.
    If <= max_tokens, returns [text] and don't do anything.
    """

    _require_model()
    
    sentences = _split_into_sentences(text, language=language, splitter=splitter)
    
    if len(sentences) == 0:
        return [text] if _text_fits_under_limit(text, limit=max_tokens) else [text[:1000]], sentences


    # Quick exit if whole text fits

    if _text_fits_under_limit(text, limit=max_tokens):
        return [text], sentences
    
    sentences = _pre_split_rogues(sentences, max_tokens=max_tokens, debug=debug)

    embs = _get_sentence_embeddings(sentences, batch_size=batch_size)
    
    return _multi_split_best_points(
        sentences,
        embs,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        window_size=window_size,
        max_prev_sentences=max_prev_sentences,
        debug=debug
    ), sentences