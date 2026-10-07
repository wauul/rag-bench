"""Reusable LangChain components; generation accepts question + passages only."""

from copy import deepcopy
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from backend.provenance import SYSTEM_PROMPT
from backend.retrieval import select_contexts


def to_documents(chunks):
    return [
        Document(
            id=c.get("chunk_id"),
            page_content=c["text"],
            metadata={k: v for k, v in c.items() if k != "text"},
        )
        for c in chunks
    ]


def to_chunks(documents):
    return [{**doc.metadata, "text": doc.page_content} for doc in documents]


class ChromaRetriever(BaseRetriever):
    """The existing normalized embeddings, query prefix, cosine index and ordering."""

    index: Any

    def _get_relevant_documents(self, query, *, run_manager):
        return to_documents(self.index.retrieve(query))


def passage_selector(config, reranker=None):
    return RunnableLambda(
        lambda value: to_documents(
            select_contexts(
                deepcopy(to_chunks(value["documents"])), config, value["question"], reranker
            )
        )
    )


def format_passages(documents):
    return "\n\n".join(
        f"[{i + 1}] {d.metadata['source']} p.{d.metadata['page']}\n{d.page_content}"
        for i, d in enumerate(documents)
    )


def generation_inputs(value):
    # An explicit allowlist: neither reference answers nor evaluation state enter the chain.
    return {"question": value["question"], "context": format_passages(value["documents"])}


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "PASSAGES:\n{context}\n\nQUESTION: {question}")]
)


def parse_answer(response):
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("Generator returned no text answer")
    return response.content


def answer_chain(llm):
    # Production uses ChatGroq directly; a callable adapter supports deterministic test doubles.
    from langchain_core.runnables import Runnable

    model = llm if isinstance(llm, Runnable) else RunnableLambda(llm.invoke)
    return RunnableLambda(generation_inputs) | ANSWER_PROMPT | model | RunnableLambda(parse_answer)
