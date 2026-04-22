"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

import uuid
from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

# Updated LangChain 1.0 Import
from langchain.agents import create_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
            answer=state.answer # Pass along existing answer state if present
        )

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        # 1. Retriever Tool (Safely defined)
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                chunk_id = meta.get("chunk_id", f"doc_{i}")
                title = meta.get("title") or meta.get("source") or chunk_id
                merged.append(f"[{chunk_id}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool.from_function(
            func=retriever_tool_fn,
            name="retriever",
            description="Fetch passages from indexed corpus to answer questions."
        )

        # 2. Wikipedia Tool 
        # FIX: WikipediaQueryRun is already a tool! Do not wrap it in another Tool()
        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        # Return the clean tools
        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        
        self._agent = create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )