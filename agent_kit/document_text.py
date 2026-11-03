"""Each document's text, the source of truth for every quote: the inbox, page images, and
text extraction through the Files API. Used by analyze_document, the case record, and the
memo rules.

Each document's text is extracted once (uploaded through the Files API, then transcribed
by the model), saved under workspace/cache/ keyed by the file's SHA-256, and reused.
Every quote is checked against that saved text (quote_on_page); record_fact refuses a
quote that is not on the stated page.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import cast

from openai import OpenAI
from openai.types.responses import Response

from . import progress
from .settings import model


_REGISTRY_LOCK = threading.Lock()
MAX_ATTEMPTS = 2
_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


class DocumentError(ValueError):
    pass


def normalize(text: str) -> str:
    return _SPACE.sub(" ", (text or "").translate(_QUOTES)).strip().lower()


def quote_on_page(quote: str, page_text: str) -> bool:
    q = normalize(quote)
    return len(q) >= 8 and q in normalize(page_text)


# ---- inbox ------------------------------------------------------------------


DOCUMENT_TYPES = {".pdf": "pdf", ".jpg": "image", ".jpeg": "image", ".png": "image"}
_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def page_count(path: Path) -> int:
    """Rendered page images if there are any (pages/<ID>-p<n>.png), else pages in the PDF, else 1."""
    rendered = len(list((path.parent / "pages").glob(f"{path.stem}-p*.png")))
    if rendered:
        return rendered
    if path.suffix.lower() == ".pdf":
        return max(1, len(_PDF_PAGE.findall(path.read_bytes())))
    return 1


def list_documents(root: Path) -> list:
    """Documents currently in workspace/inbox/: id (the file name), file, modality (pdf | image), pages, sha256."""
    inbox = root / "workspace/inbox"
    if not inbox.exists():
        return []
    return [
        {
            "id": path.stem,
            "file": path.name,
            "modality": DOCUMENT_TYPES[path.suffix.lower()],
            "pages": page_count(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(inbox.iterdir())
        if path.is_file() and path.suffix.lower() in DOCUMENT_TYPES
    ]


def get_document(root: Path, document_id: str) -> dict:
    for document in list_documents(root):
        if document["id"] == document_id:
            return document
    raise DocumentError(f"document {document_id!r} is not in the inbox")


def page_image(root: Path, document_id: str, page: int) -> Path:
    """The review platform's 100 dpi render of one page."""
    get_document(root, document_id)
    path = root / "workspace/inbox/pages" / f"{document_id}-p{page}.png"
    if not path.is_file():
        raise DocumentError(f"no page image for {document_id} page {page}")
    return path


# ---- text extraction (Files API) ----------------------------------------------


def ask(
    client: OpenAI,
    content: list,
    *,
    root: Path | None = None,
    run_id: str = "",
    task: str = "transcribe",
) -> str:
    """One JSON-mode model call. Returns the output text."""
    step_entry = (
        progress.start_step(root, run_id, task=task, model=model(), summary="Working…")
        if root is not None else None
    )
    t0 = time.perf_counter()
    response = client.responses.create(
        model=model(),
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_object"}},
    )
    if root is not None:
        usage = getattr(response, "usage", None)
        progress.finish_step(
            root, run_id, step_entry["step"] if step_entry else 0,
            duration_ms=(time.perf_counter() - t0) * 1000,
            summary=f"{task} model call",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
    return cast(Response, response).output_text


def cache(root: Path) -> Path:
    path = root / "workspace/cache"
    (path / "text").mkdir(parents=True, exist_ok=True)
    (path / "analysis").mkdir(parents=True, exist_ok=True)
    return path


def cached_text(root: Path, document_id: str) -> dict | None:
    """The saved transcription for a document, or None if not extracted yet."""
    document = get_document(root, document_id)
    path = cache(root) / "text" / f"{document['sha256']}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def page_text(root: Path, document_id: str, page: int) -> str | None:
    text = cached_text(root, document_id)
    if text is None:
        return None
    for entry in text["pages"]:
        if entry["page"] == page:
            return entry["text"]
    return ""


def _file_id(client: OpenAI, root: Path, document: dict) -> str:
    registry_path = cache(root) / "files.json"

    def registry() -> dict:
        return json.loads(registry_path.read_text()) if registry_path.is_file() else {}

    with _REGISTRY_LOCK:
        known = registry().get(document["sha256"])
    if known:
        return known
    with (root / "workspace/inbox" / document["file"]).open("rb") as stream:
        file_id = client.files.create(file=stream, purpose="user_data").id
    with _REGISTRY_LOCK:
        current = registry()
        current[document["sha256"]] = file_id
        registry_path.write_text(json.dumps(current, indent=2))
    return file_id


TRANSCRIBE_PROMPT = (
    "Transcribe document {doc_id} ({pages} page(s)) exactly as written, page by page. "
    "Include headers, table cells, chart labels and values, handwriting, and stamps. "
    "Do not summarize, correct, or reorder. Write [illegible] for text you cannot read. "
    'Return only JSON: {{"pages": [{{"page": 1, "text": "..."}}]}} with one entry per page.'
)


def extract_text(
    client: OpenAI, root: Path, document_id: str, *, run_id: str = ""
) -> dict:
    """Extract once, save under workspace/cache/text/<sha>.json (see the module docstring)."""
    existing = cached_text(root, document_id)
    if existing is not None:
        return existing
    document = get_document(root, document_id)
    file_id = _file_id(client, root, document)
    prompt = TRANSCRIBE_PROMPT.format(doc_id=document["id"], pages=document["pages"])
    error = ""
    for _attempt in range(MAX_ATTEMPTS):
        output = ask(
            client,
            [
                {
                    "type": "input_text",
                    "text": prompt
                    + (f"\nYour previous output was invalid: {error}" if error else ""),
                },
                {"type": "input_file", "file_id": file_id},
            ],
            root=root, run_id=run_id, task="transcribe",
        )
        try:
            parsed = json.loads(output)
            pages = [
                {"page": int(p["page"]), "text": str(p["text"])}
                for p in parsed["pages"]
            ]
            if not pages:
                raise ValueError("no pages")
            break
        except (ValueError, KeyError, TypeError) as problem:
            error = str(problem)
    else:
        raise DocumentError(f"could not extract text from {document_id}: {error}")
    return _save_text(root, document, pages, "files_api", file_id)


def _save_text(
    root: Path,
    document: dict,
    pages: list,
    source: str,
    file_id: str | None = None,
) -> dict:
    record = {
        "document_id": document["id"],
        "sha256": document["sha256"],
        "source": source,
        "file_id": file_id,
        "pages": pages,
    }
    (cache(root) / "text" / f"{document['sha256']}.json").write_text(
        json.dumps(record, indent=1)
    )
    return record
