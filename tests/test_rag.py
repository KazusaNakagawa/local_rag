import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import HumanMessage, AIMessage

from rag import format_docs, contextualize_query, stream_answer, _merge_results, load_resources


# ── load_resources ────────────────────────────────────────

def test_load_resources_requires_allow_deserialization():
    """allow_deserialization=False（デフォルト）では ValueError を送出する。"""
    with pytest.raises(ValueError, match="allow_deserialization"):
        load_resources("/any/path")


# ── _merge_results ─────────────────────────────────────────

def test_merge_results_deduplicates_same_source_and_content():
    """同じソースと本文を持つドキュメントは重複除去される。"""
    doc = Document(page_content="text", metadata={"source": "a.md"})
    duplicate = Document(page_content="text", metadata={"source": "a.md"})
    result = _merge_results([doc], [duplicate], top_k=10)
    assert len(result) == 1


def test_merge_results_keeps_same_content_different_source():
    """同じ本文でもソースが異なれば別ドキュメントとして保持する。"""
    doc1 = Document(page_content="text", metadata={"source": "a.md"})
    doc2 = Document(page_content="text", metadata={"source": "b.md"})
    result = _merge_results([doc1], [doc2], top_k=10)
    assert len(result) == 2


def test_merge_results_bm25_first():
    """BM25 の結果が先頭に来る。"""
    bm25_doc = Document(page_content="bm25", metadata={"source": "a.md"})
    faiss_doc = Document(page_content="faiss", metadata={"source": "b.md"})
    result = _merge_results([bm25_doc], [faiss_doc], top_k=10)
    assert result[0].page_content == "bm25"


def test_merge_results_respects_top_k():
    """top_k で上限を超えないことを確認。"""
    docs = [Document(page_content=f"doc{i}", metadata={"source": f"{i}.md"}) for i in range(6)]
    result = _merge_results(docs[:3], docs[3:], top_k=4)
    assert len(result) == 4


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
