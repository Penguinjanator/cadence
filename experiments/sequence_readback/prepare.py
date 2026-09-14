"""Freeze small, disjoint book excerpts from the existing public-domain corpus cache."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "cadence-author" / "author"


def main() -> None:
    spec = importlib.util.spec_from_file_location("author_corpus", SOURCE / "corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (HERE / "data").mkdir(exist_ok=True)
    metadata = {
        "alphabet": module.ALPHABET,
        "normalizer_sha256": hashlib.sha256((SOURCE / "corpus.py").read_bytes()).hexdigest(),
        "splits": {},
    }
    for split, book, title, count in (
        ("train", 36, "The War of the Worlds", 60000),
        ("validation", 61963, "We", 10000),
        ("test", 21970, "The Scarlet Plague", 10000),
    ):
        path = SOURCE / "data" / f"pg{book}.txt"
        raw = path.read_bytes()
        offset = 11000 if split == "test" else 1000
        text = module.normalise(module.body(raw.decode("utf-8", errors="replace")))[
            offset : offset + count
        ]
        data = text.encode("utf-8")
        (
            HERE / "data" / ("test_confirmation.txt" if split == "test" else f"{split}.txt")
        ).write_bytes(data)
        metadata["splits"][split] = {
            "book": title,
            "gutenberg_id": book,
            "url": module.URL.format(id=book),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "normalized_offset": offset,
            "characters": len(text),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    old_manifest = HERE / "data" / "manifest.json"
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if "pilot_test" in old:
            metadata["pilot_test"] = old["pilot_test"]
    (HERE / "data" / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
