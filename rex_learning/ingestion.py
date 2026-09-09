from __future__ import annotations

import hashlib
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .engine import LearningEngine
from .learner import Learner
from .capability_discovery import discover_capabilities, discover_capabilities_from_units


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.heading: str | None = None
        self.sections: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3"}:
            self.heading = ""

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self.heading is not None:
            self.heading = f"{self.heading} {text}".strip()
        else:
            self.parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3"} and self.heading:
            self.sections.append({"heading": self.heading[:200], "text": ""})
            self.heading = None
        elif tag in {"p", "li"} and self.parts:
            text = self.parts.pop()
            if self.sections:
                self.sections[-1]["text"] = f"{self.sections[-1]['text']} {text}".strip()[:4000]
            else:
                self.sections.append({"heading": "untitled", "text": text[:4000]})


def _manifest_hash(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, data in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _plain_sections(text: str, source_path: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if body:
                sections.append({"source_path": source_path, "heading": heading or "untitled", "text": " ".join(body)[:4000]})
                body = []
            heading = stripped.lstrip("#").strip()[:200] or "untitled"
        else:
            body.append(stripped)
    if body:
        sections.append({"source_path": source_path, "heading": heading or "untitled", "text": " ".join(body)[:4000]})
    return sections


def _sections_for_file(data: bytes, source_path: str) -> list[dict[str, Any]]:
    """Decode one supported directory member using its declared format."""
    suffix = Path(source_path).suffix.casefold()
    if suffix in {".html", ".htm", ".xhtml"}:
        parser = _TextParser()
        parser.feed(data.decode("utf-8", errors="replace"))
        return [
            {"source_path": source_path, **section}
            for section in parser.sections
            if section.get("text")
        ]
    return _plain_sections(data.decode("utf-8", errors="replace"), source_path)


def _text_ingestion(
    engine: LearningEngine,
    curriculum_id: str,
    path: Path,
    *,
    chunk_chars: int,
    discovery_packet_chars: int,
    learner: Learner | None,
    educational_objective: str | None,
) -> dict[str, Any]:
    if path.is_dir():
        files = [
            (file.relative_to(path).as_posix(), file.read_bytes())
            for file in sorted(path.rglob("*"))
            if file.is_file() and not file.is_symlink() and file.suffix.casefold() in {".md", ".markdown", ".txt", ".html", ".htm", ".xhtml"}
        ]
        if not files:
            raise ValueError("source directory contains no supported instructional files")
        source_kind = "directory"
        source_hash = _manifest_hash(files)
        source_title = path.name
        sections = [section for relative, data in files for section in _sections_for_file(data, relative)]
    else:
        data = path.read_bytes()
        source_kind = "markdown" if path.suffix.casefold() in {".md", ".markdown"} else "text"
        source_hash = hashlib.sha256(data).hexdigest()
        source_title = path.name
        sections = _plain_sections(data.decode("utf-8", errors="replace"), path.name)
    source = engine.add_source(curriculum_id, source_title, str(path), kind=source_kind, content_hash=source_hash)
    chunks = [
        {"source_path": section["source_path"], "heading": section["heading"], "text": section["text"][start:start + chunk_chars], "source_ref": f"{section['source_path']}#{section['heading']}:{start}"}
        for section in sections
        for start in range(0, len(section["text"]), chunk_chars)
    ]
    notes: list[str] = []
    candidates: list[dict[str, str]] = []
    for chunk in chunks:
        claim = re.sub(r"\s+", " ", chunk["text"]).strip()
        note = engine.record_study_note(curriculum_id, source["id"], {"kind": "extracted_section", "claim": claim, "conditions": [chunk["source_ref"]]})
        notes.append(note["id"])
        if any(word in claim.casefold() for word in ("procedure", "classify", "step", "must")):
            candidates.append({"claim": claim[:1000], "source_ref": chunk["source_ref"], "note_id": note["id"]})
    result: dict[str, Any] = {"schema": "rex-learning-source-ingestion-v1", "source": source, "sections": sections, "chunks": chunks, "study_note_ids": notes, "candidate_skills": candidates}
    if learner is not None or educational_objective is not None:
        if learner is None or not educational_objective or not educational_objective.strip():
            raise ValueError("learner and educational_objective must be provided together")
        result["capability_discovery"] = discover_capabilities(
            engine=engine, learner=learner, source_title=source_title,
            source_text="\n\n".join(f"## {section['heading']}\n{section['text']}" for section in sections),
            educational_objective=educational_objective, curriculum_id=curriculum_id,
            source_locator=str(path), source_refs=[chunk["source_ref"] for chunk in chunks], source_id=source["id"],
            source_hash=source["content_hash"],
            packet_chars=discovery_packet_chars,
        )
    return result


def ingest_source(
    engine: LearningEngine,
    curriculum_id: str,
    path: Path,
    *,
    chunk_chars: int = 1200,
    discovery_packet_chars: int = 12000,
    learner: Learner | None = None,
    educational_objective: str | None = None,
) -> dict[str, Any]:
    """Ingest EPUB, text, Markdown, or a deterministic text directory."""
    path = Path(path)
    if path.is_file() and path.suffix.casefold() == ".epub":
        return ingest_epub(engine, curriculum_id, path, chunk_chars=chunk_chars, learner=learner, educational_objective=educational_objective)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.suffix.casefold() not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"unsupported source format: {path.suffix or '<none>'}")
    return _text_ingestion(engine, curriculum_id, path, chunk_chars=chunk_chars, discovery_packet_chars=discovery_packet_chars, learner=learner, educational_objective=educational_objective)


def ingest_epub(
    engine: LearningEngine,
    curriculum_id: str,
    path: Path,
    *,
    chunk_chars: int = 1200,
    discovery_packet_chars: int = 12000,
    learner: Learner | None = None,
    educational_objective: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    source = engine.add_source(curriculum_id, path.name, str(path), kind="epub", content_hash=hashlib.sha256(data).hexdigest())
    sections: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as book:
        for name in sorted(book.namelist()):
            if not name.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            parser = _TextParser()
            parser.feed(book.read(name).decode("utf-8", errors="replace"))
            for section in parser.sections:
                if section["text"]:
                    sections.append({"source_path": name, **section})
    chunks = []
    for section in sections:
        text = section["text"]
        for start in range(0, len(text), chunk_chars):
            chunks.append({"source_path": section["source_path"], "heading": section["heading"], "text": text[start:start + chunk_chars], "source_ref": f"{section['source_path']}#{section['heading']}:{start}"})
    notes = []
    candidates = []
    for chunk in chunks:
        claim = re.sub(r"\s+", " ", chunk["text"]).strip()
        note = engine.record_study_note(curriculum_id, source["id"], {"kind": "extracted_section", "claim": claim, "conditions": [chunk["source_ref"]]})
        notes.append(note["id"])
        if any(word in claim.casefold() for word in ("procedure", "classify", "step", "must")):
            candidates.append({"claim": claim[:1000], "source_ref": chunk["source_ref"], "note_id": note["id"]})
    result: dict[str, Any] = {"schema": "rex-learning-epub-ingestion-v1", "source": source, "sections": sections, "chunks": chunks, "study_note_ids": notes, "candidate_skills": candidates}
    if learner is not None or educational_objective is not None:
        if learner is None or not educational_objective or not educational_objective.strip():
            raise ValueError("learner and educational_objective must be provided together")
        result["capability_discovery"] = discover_capabilities_from_units(
            engine=engine,
            learner=learner,
            source_title=path.name,
            source_units=[
                {"unit_id": chunk["source_ref"], "title": chunk["heading"], "text": chunk["text"], "source_ref": chunk["source_ref"]}
                for chunk in chunks
            ],
            educational_objective=educational_objective,
            curriculum_id=curriculum_id,
            source_id=source["id"],
            packet_chars=discovery_packet_chars,
        )
    return result
