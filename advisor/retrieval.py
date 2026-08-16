"""Embedding retrieval over filing text, LangChain.

This replaces a hand-written passage scorer. The old version jumped to headings
like "Management's Discussion and Analysis" and ranked candidates by counting
financial vocabulary, which worked but broke in instructive ways: the same
heading appears in the table of contents, in cross-references, and in the
section itself, so picking the wrong one fed the model navigation text.

Semantic retrieval removes the heading problem rather than patching it. A table
of contents entry does not resemble a sentence about margin compression, so it
loses on cosine distance without needing a rule that says so.

There is no silent fallback to the old scorer. Retrieval quality decides what
evidence the model sees, so degrading it quietly would change every number in
the backtest while the run still looked fine. This project has been bitten twice
by failures that returned plausible values instead of raising, so this one
raises.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

# What we are looking for in a filing. Retrieval is only as good as the question
# asked of it, and the question here is fixed: the model's job downstream is to
# rank peers against each other, so the evidence it needs is the part of the
# filing describing how the business actually moved.
# Phrased as movement rather than as topics, which is not cosmetic. An earlier
# version listed subjects ("forward guidance, competitive position, material
# business risk") and retrieved the risk-factor and forward-looking-statements
# boilerplate that names those subjects without reporting anything: "Actual
# future results...", "Changes in trade, monetary and fiscal policies". Asking
# for the change itself retrieves the sentences carrying figures instead.
QUERY = (
    "net sales increased or decreased compared to the prior year period, "
    "gross margin change, operating expenses, segment results, demand outlook"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 3

# Filings are stored whitespace-collapsed, so there are no paragraph breaks left
# to split on. Sentence boundaries are the next best seam: a chunk that stops
# mid-clause embeds badly and reads badly.
SEPARATORS = [". ", "; ", ", ", " ", ""]


class EmbeddingsUnavailable(RuntimeError):
    """Raised when the embedding model cannot be loaded."""


@lru_cache(maxsize=1)
def _default_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise EmbeddingsUnavailable(f"langchain-huggingface not installed: {exc}") from exc
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as exc:  # pragma: no cover - download or torch failure
        raise EmbeddingsUnavailable(f"could not load {MODEL_NAME}: {exc}") from exc


@lru_cache(maxsize=1)
def _splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS
    )


def config_fingerprint() -> str:
    """Short hash of everything that changes retrieval output.

    Cached excerpts are keyed by this, so editing the query or the chunk size
    invalidates them instead of silently mixing two retrieval schemes across one
    backtest.
    """
    blob = f"{QUERY}|{MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{TOP_K}"
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def retrieve(text: str, max_chars: int = 2000, k: int = TOP_K, embeddings=None) -> str:
    """Top-k chunks of `text` most relevant to QUERY, in document order.

    Document order matters. Ranked order interleaves an outlook sentence from
    page 40 with a revenue line from page 6, and the model reads the result as
    one passage.
    """
    if not text or not text.strip():
        return ""

    chunks = _splitter().split_text(text)
    if not chunks:
        return ""
    if len(chunks) <= k:
        return " ... ".join(chunks)[:max_chars]

    from langchain_core.vectorstores import InMemoryVectorStore

    embeddings = embeddings or _default_embeddings()
    # Chunk index is carried through as metadata so the winners can be put back
    # into the order they appear in the filing.
    store = InMemoryVectorStore.from_texts(
        chunks, embeddings, metadatas=[{"i": i} for i in range(len(chunks))]
    )
    hits = store.similarity_search(QUERY, k=k)
    ordered = sorted(hits, key=lambda d: d.metadata.get("i", 0))
    return " ... ".join(d.page_content for d in ordered)[:max_chars]


def retrieve_cached(text: str, cache_path: Path, max_chars: int = 2000, **kwargs) -> str:
    """`retrieve`, memoised on disk.

    Embedding a 10-K is a few hundred chunks and the backtest reads the same
    filings on every re-run. The fingerprint is in the filename, so a changed
    query writes a new file rather than returning a stale one.
    """
    path = cache_path.with_suffix(f".{config_fingerprint()}.{max_chars}.rag")
    if path.exists():
        return path.read_text()
    excerpt = retrieve(text, max_chars=max_chars, **kwargs)
    if excerpt:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(excerpt)
    return excerpt
