import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import HumanMessage, AIMessage

from rag import format_docs, contextualize_query, stream_answer


# ── format_docs ────────────────────────────────────────────

def test_format_docs_empty():
    assert format_docs([]) == ""


def test_format_docs_includes_basename():
    doc = Document(page_content="本文", metadata={"source": "/path/to/note.md"})
    result = format_docs([doc])
    assert "note.md" in result
    assert "本文" in result


def test_format_docs_multiple_docs():
    docs = [
        Document(page_content="A", metadata={"source": "a.md"}),
        Document(page_content="B", metadata={"source": "b.md"}),
    ]
    result = format_docs(docs)
    assert "Source 1" in result
    assert "Source 2" in result
    assert "a.md" in result
    assert "b.md" in result


# ── contextualize_query ────────────────────────────────────

def test_contextualize_query_no_history_skips_llm():
    """履歴なしの場合は LLM を呼ばずそのまま返す。"""
    llm = FakeListLLM(responses=["LLM が呼ばれたら失敗"])
    result = contextualize_query(llm, "元の質問", [])
    assert result == "元の質問"


def test_contextualize_query_with_history_calls_llm():
    """履歴ありの場合は LLM で言い換えた質問を返す。"""
    llm = FakeListLLM(responses=["言い換えた質問"])
    chat_history = [HumanMessage(content="前の質問"), AIMessage(content="前の回答")]
    result = contextualize_query(llm, "それについて詳しく", chat_history)
    assert result == "言い換えた質問"


# ── stream_answer ──────────────────────────────────────────

def test_stream_answer_yields_chunks():
    llm = FakeListLLM(responses=["回答テキスト"])
    docs = [Document(page_content="コンテキスト", metadata={"source": "test.md"})]
    chunks = list(stream_answer(llm, "質問", [], docs))
    assert "".join(chunks) == "回答テキスト"


def test_stream_answer_with_history():
    llm = FakeListLLM(responses=["履歴あり回答"])
    docs = [Document(page_content="ctx", metadata={"source": "n.md"})]
    chat_history = [HumanMessage(content="Q1"), AIMessage(content="A1")]
    chunks = list(stream_answer(llm, "Q2", chat_history, docs))
    assert "".join(chunks) == "履歴あり回答"
