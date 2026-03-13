from __future__ import annotations

import argparse
import json

from .service import RAGSystem
from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs into the local ChromaDB store.")
    parser.add_argument("--force", action="store_true", help="Rebuild all indexed documents.")
    args = parser.parse_args()

    try:
        rag = RAGSystem(Settings.from_env())
        summary = rag.sync_documents(force=args.force)
        print(json.dumps(summary, indent=2))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:
        raise SystemExit(f"Indexing failed: {exc}") from exc
