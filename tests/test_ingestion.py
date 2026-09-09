from pathlib import Path

import pytest

from rex_learning import LearningEngine, LearningError, LearningStore, ingest_source


def test_directory_source_is_deterministic_and_rejects_changed_content(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "01.md").write_text("# One\nFirst procedure.\n", encoding="utf-8")
    (corpus / "nested").mkdir()
    (corpus / "nested/02.txt").write_text("Second procedure.\n", encoding="utf-8")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("corpus", "Find transferable software procedures")

    first = ingest_source(engine, curriculum["id"], corpus)
    second = ingest_source(engine, curriculum["id"], corpus)

    assert first["schema"] == "rex-learning-source-ingestion-v1"
    assert first["source"]["kind"] == "directory"
    assert first["source"]["content_hash"] == second["source"]["content_hash"]
    assert first["source"]["id"] == second["source"]["id"]
    assert [chunk["source_path"] for chunk in first["chunks"]] == ["01.md", "nested/02.txt"]
    assert all("#" in chunk["source_ref"] for chunk in first["chunks"])

    (corpus / "nested/02.txt").write_text("Changed procedure.\n", encoding="utf-8")
    with pytest.raises(LearningError, match="source locator content hash cannot change"):
        ingest_source(engine, curriculum["id"], corpus)


def test_unsupported_regular_source_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"not a PDF parser input")
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("guide.pdf", "Find transferable software procedures")

    with pytest.raises(ValueError, match=r"unsupported source format: \.pdf"):
        ingest_source(engine, curriculum["id"], source)


def test_directory_source_ingests_html_members(tmp_path: Path) -> None:
    corpus = tmp_path / "html-corpus"
    corpus.mkdir()
    (corpus / "chapter.html").write_text(
        "<h1>Testing procedure</h1><p>Isolate the unit and verify behavior with a focused test.</p>",
        encoding="utf-8",
    )
    engine = LearningEngine(LearningStore(tmp_path / "learning"))
    curriculum = engine.create_curriculum("html-corpus", "Find transferable software procedures")

    result = ingest_source(engine, curriculum["id"], corpus)

    assert result["source"]["kind"] == "directory"
    assert result["chunks"][0]["source_path"] == "chapter.html"
    assert "Isolate the unit" in result["chunks"][0]["text"]
